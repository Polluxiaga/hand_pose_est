#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]

from match_geometry_core import projected_uv_to_mask
from init_seed_core import (
    SeedCandidate,
    build_sam3d_seed_candidates as _build_sam3d_seed_candidates,
    build_tapir_seed_candidates as _build_seed_candidates,
    prior_keep_debug as _prior_keep_debug,
    reconstruct_init_transform as _reconstruct_tapir_transform,
    registration_working_target_points as _registration_working_target_points,
)
from pcd_registration_core import (
    CameraIntrinsics,
    RegistrationConfig,
    copy_pcd,
    local_hypothesis_search,
    refine_with_colored_icp,
    refine_with_gicp,
    visibility_aware_score,
)
from tapir_match_core import (
    TapirMatchConfig,
    draw_tapir_matches,
    estimate_tapir_init_transform,
)
from sam3d_match_core import (
    DEFAULT_SAM3D_PICK_PROMPT,
    DEFAULT_SAM3D_URL,
    Sam3dMatchConfig,
    estimate_sam3d_init_transform,
    evaluate_sam3d_prior,
)

DEFAULT_DOF_MATCH_DIR = "dof_match"
DEFAULT_OUTPUT_DIR = "dof_batch_results"
DEFAULT_TAPIR_HOST = os.getenv("TAPIR_HOST", "192.168.20.212")
DEFAULT_TAPIR_PORT = int(os.getenv("TAPIR_PORT", "19812"))


@dataclass
class ReEstimationResult:
    success: bool
    mode: str
    T_registration: np.ndarray
    T_slip: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


def _emit_runtime_log(log_fn, message: str) -> None:
    if not callable(log_fn):
        return
    stamp = time.strftime("%H:%M:%S")
    log_fn(f"[{stamp}] {message}")


def depth_to_meters(depth_data: np.ndarray, target_hw: tuple[int, int] | None = None) -> np.ndarray:
    if depth_data.ndim == 3 and depth_data.shape[2] == 3:
        depth_m = depth_data[:, :, 2].astype(np.float32)
    elif depth_data.ndim == 2:
        depth_m = depth_data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth shape: {depth_data.shape}")

    if np.issubdtype(depth_data.dtype, np.integer):
        depth_m *= 0.001

    invalid = ~np.isfinite(depth_m) | (depth_m <= 0.0)
    depth_m[invalid] = 0.0

    if target_hw is not None:
        target_h, target_w = target_hw
        if depth_m.shape[:2] != (target_h, target_w):
            depth_m = cv2.resize(depth_m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return depth_m


def _read_json_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def _find_dof_log_roots(root: str | os.PathLike[str]) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    roots: list[Path] = []
    if (root_path / "teach_dof").is_dir() and (root_path / "teach_dof_replay").is_dir():
        roots.append(root_path)
    else:
        for replay_root in sorted(root_path.rglob("teach_dof_replay")):
            log_root = replay_root.parent
            if replay_root.is_dir() and (log_root / "teach_dof").is_dir():
                roots.append(log_root)
    deduped: list[Path] = []
    seen: set[str] = set()
    for root_item in roots:
        key = str(root_item)
        if key not in seen:
            deduped.append(root_item)
            seen.add(key)
    return deduped


def _resolve_dof_memory_dir(replay_dir: Path, log_root: Path) -> tuple[Path | None, str]:
    request_data = _read_json_file(replay_dir / "request.json") if (replay_dir / "request.json").is_file() else {}
    response_data = _read_json_file(replay_dir / "response.json") if (replay_dir / "response.json").is_file() else {}
    memory_dir_raw = str(request_data.get("memory_dir") or response_data.get("memory_dir") or "").strip()
    if not memory_dir_raw:
        return None, ""

    memory_name = os.path.basename(os.path.normpath(memory_dir_raw))
    candidates = [
        Path(memory_dir_raw).expanduser(),
        log_root / "teach_dof" / memory_name,
        replay_dir.parent.parent / "teach_dof" / memory_name,
    ]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.is_dir():
            return resolved, memory_dir_raw
    return None, memory_dir_raw


def _iter_dof_pairs(root: str | os.PathLike[str]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for log_root in _find_dof_log_roots(root):
        replay_root = log_root / "teach_dof_replay"
        for replay_dir in sorted(path for path in replay_root.iterdir() if path.is_dir()):
            if not (replay_dir / "request.json").is_file():
                continue
            before_dir, memory_dir_raw = _resolve_dof_memory_dir(replay_dir, log_root)
            pairs.append(
                {
                    "log_root": log_root,
                    "before_dir": before_dir,
                    "replay_dir": replay_dir,
                    "memory_dir": memory_dir_raw,
                }
            )
    return pairs


def _load_dof_intrinsics(artifact_dir: Path) -> dict[str, float]:
    intr_path = artifact_dir / "intrinsics.json"
    if intr_path.is_file():
        data = _read_json_file(intr_path)
        values = data.get("values", data)
    else:
        data = _read_json_file(artifact_dir / "request.json")
        values = data.get("intrinsics", {})
    if not isinstance(values, dict):
        raise ValueError(f"Invalid intrinsics in {artifact_dir}")
    return {
        "fx": float(values["fx"]),
        "fy": float(values["fy"]),
        "cx": float(values["cx"]),
        "cy": float(values["cy"]),
        "baseline": float(values.get("baseline", 0.0)),
        "width": float(values.get("width", 0.0)),
        "height": float(values.get("height", 0.0)),
    }


def _load_dof_extrinsics(artifact_dir: Path, request_data: dict[str, Any]) -> dict[str, Any]:
    path = artifact_dir / "extrinsics.json"
    data: dict[str, Any] = {}
    if path.is_file():
        data = _read_json_file(path)
    elif isinstance(request_data.get("extrinsics"), dict):
        data = dict(request_data.get("extrinsics") or {})
    if "rotation" not in data or "shift" not in data:
        return {}
    rotation = np.asarray(data.get("rotation"), dtype=np.float64)
    shift = np.asarray(data.get("shift"), dtype=np.float64)
    if rotation.shape != (3, 3) or shift.shape != (3,):
        return {}
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(shift)):
        return {}
    return {
        "rotation": rotation.astype(np.float32, copy=False),
        "shift": shift.astype(np.float32, copy=False),
    }


def _load_dof_object_prompt(artifact_dir: Path, is_replay: bool, request_data: dict[str, Any]) -> str:
    request_prompt = str(request_data.get("object_prompt") or "").strip()
    if request_prompt:
        return request_prompt
    seg_name = "current_object_segment.json" if is_replay else "object_segment.json"
    seg_data = _load_json_if_present(artifact_dir / seg_name)
    prompt = str(seg_data.get("prompt") or "").strip()
    if prompt:
        return prompt
    qwen_debug = seg_data.get("qwen_debug") or {}
    if isinstance(qwen_debug, dict):
        prompt = str(qwen_debug.get("prompt") or "").strip()
        if prompt:
            return prompt
    return DEFAULT_SAM3D_PICK_PROMPT


def _mask_from_pixels(pixel_path: Path, height: int, width: int) -> np.ndarray:
    pixels = np.asarray(np.load(str(pixel_path)))
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"Expected Nx2 pixel array: {pixel_path}, got {pixels.shape}")
    xy = np.rint(pixels).astype(np.int64)
    x = xy[:, 0]
    y = xy[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[y[inside], x[inside]] = 255
    return mask


def _load_dof_mask(artifact_dir: Path, height: int, width: int, is_replay: bool) -> np.ndarray:
    name = "current_object_selected_pixels_xy.npy" if is_replay else "object_selected_pixels_xy.npy"
    path = artifact_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing object pixel mask: {path}")
    return _mask_from_pixels(path, height, width)


def _load_json_if_present(path: Path) -> dict[str, Any]:
    return _read_json_file(path) if path.is_file() else {}


def _artifact_object_prefix(is_replay: bool) -> str:
    return "current_object_selected" if is_replay else "object_selected"


def _load_cached_object_arrays(artifact_dir: Path, is_replay: bool) -> dict[str, Any]:
    prefix = _artifact_object_prefix(is_replay)
    xyz_path = artifact_dir / f"{prefix}_xyz.npy"
    rgb_path = artifact_dir / f"{prefix}_rgb.npy"
    pixel_path = artifact_dir / f"{prefix}_pixels_xy.npy"
    if not (xyz_path.is_file() and rgb_path.is_file() and pixel_path.is_file()):
        return {}

    xyz = np.asarray(np.load(str(xyz_path)), dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(np.load(str(rgb_path)), dtype=np.float32).reshape(-1, 3)
    pixels_xy = np.asarray(np.load(str(pixel_path)))
    if pixels_xy.ndim != 2 or pixels_xy.shape[1] != 2:
        raise ValueError(f"Expected Nx2 pixel array: {pixel_path}, got {pixels_xy.shape}")

    n = min(len(xyz), len(rgb), len(pixels_xy))
    xyz = xyz[:n]
    rgb = rgb[:n]
    pixels_xy = np.asarray(pixels_xy[:n], dtype=np.float32)
    finite = np.isfinite(xyz).all(axis=1)
    if np.issubdtype(rgb.dtype, np.number):
        finite &= np.isfinite(rgb).all(axis=1)
    xyz = xyz[finite]
    rgb = rgb[finite]
    pixels_xy = pixels_xy[finite]
    return {
        "point_xyz": xyz,
        "point_rgb": np.clip(rgb, 0.0, 1.0).astype(np.float32),
        "pixels_xy": pixels_xy,
        "prefix": prefix,
        "points": int(len(xyz)),
        "xyz_path": str(xyz_path),
        "rgb_path": str(rgb_path),
        "pixel_path": str(pixel_path),
    }


def _compute_axes_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return np.eye(3, dtype=np.float64), pts.mean(axis=0) if len(pts) else np.zeros(3, dtype=np.float64)
    center = pts.mean(axis=0)
    pts0 = pts - center
    _, _, vh = np.linalg.svd(pts0, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes.astype(np.float64), center.astype(np.float64)


def _load_reference_obb(artifact_dir: Path, response_data: dict[str, Any], cached_arrays: dict[str, Any]) -> dict[str, Any]:
    obb_data: dict[str, Any] = {}
    object_data = response_data.get("object")
    if isinstance(object_data, dict):
        fast_obb = object_data.get("fast_obb")
        if isinstance(fast_obb, dict):
            obb_data = fast_obb
    if not obb_data:
        path = artifact_dir / "object_fast_obb.json"
        if path.is_file():
            obb_data = _read_json_file(path)

    if obb_data:
        axes = np.asarray(obb_data.get("axes", np.eye(3)), dtype=np.float64).reshape(3, 3)
        center = np.asarray(obb_data.get("center_m", np.zeros(3)), dtype=np.float64).reshape(3)
        if np.linalg.det(axes) < 0.0:
            axes[:, -1] *= -1.0
        return {
            "axes": axes,
            "center_m": center,
        }

    points = np.asarray(cached_arrays.get("point_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float64)
    axes, center = _compute_axes_from_points(points)
    return {
        "axes": axes,
        "center_m": center,
    }


def _load_selected_replay_pose(artifact_dir: Path) -> dict[str, Any]:
    return _load_json_if_present(artifact_dir / "selected_replay_pose.json")


def _build_pointcloud_from_cached(sample: dict[str, Any]) -> Any:
    if o3d is None:
        raise RuntimeError("open3d is required")
    points = np.asarray(sample.get("object_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float64)
    colors = np.asarray(sample.get("object_rgb", np.zeros((0, 3), dtype=np.float32)), dtype=np.float64)
    if len(points) == 0:
        return o3d.geometry.PointCloud()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if len(colors) == len(points):
        pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def _build_pointcloud_from_arrays(points_xyz: np.ndarray, points_rgb: np.ndarray | None = None) -> Any:
    if o3d is None:
        raise RuntimeError("open3d is required")
    points = np.asarray(points_xyz, dtype=np.float64).reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    if len(points) == 0:
        return pcd
    pcd.points = o3d.utility.Vector3dVector(points)
    if points_rgb is not None:
        colors = np.asarray(points_rgb, dtype=np.float64).reshape(-1, 3)
        if len(colors) == len(points):
            pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))
    return pcd


def _prepare_registration_pcd(sample: dict[str, Any], cfg: RegistrationConfig) -> tuple[Any, dict[str, Any]]:
    object_xyz = np.asarray(sample.get("object_xyz", np.zeros((0, 3), dtype=np.float32)))
    object_rgb = np.asarray(sample.get("object_rgb", np.zeros((0, 3), dtype=np.float32)))
    if len(object_xyz) == 0:
        raise ValueError("Cached object_xyz is required for pose re-estimation.")
    pcd = _reestimate_pcd_normals(_build_pointcloud_from_cached(sample), cfg)
    return pcd, {
        "source": "cached_object_xyz",
        "input_points": int(len(object_xyz)),
        "used_points": int(len(pcd.points)),
        "representative_points": int(len(pcd.points)),
        "applied": False,
        "strategy": "cached_exact",
        "voxel_size_m": 0.0,
        "raw_rgb_points": int(len(object_rgb)),
        "preprocess_mode": "normals_only",
    }


def _reestimate_pcd_normals(pcd: Any, cfg: RegistrationConfig) -> Any:
    if o3d is None or len(pcd.points) == 0:
        return pcd
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.normal_radius,
            max_nn=cfg.normal_max_nn,
        )
    )
    pcd.normalize_normals()
    return pcd


def _bbox_xyxy_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask_bool = np.asarray(mask) > 0
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _clip_xyxy(bbox: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
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


def _crop_rgb_mask_dense(
    rgb: np.ndarray,
    mask: np.ndarray,
    dense_map_cam: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = _clip_xyxy(tuple(int(v) for v in bbox_xyxy), int(rgb.shape[1]), int(rgb.shape[0]))
    return (
        np.asarray(rgb[y1 : y2 + 1, x1 : x2 + 1], dtype=np.uint8).copy(),
        np.asarray(mask[y1 : y2 + 1, x1 : x2 + 1]).copy(),
        np.asarray(dense_map_cam[y1 : y2 + 1, x1 : x2 + 1], dtype=np.float32).copy(),
    )


def _cap_registration_working_pcd(
    pcd: Any,
    cfg: RegistrationConfig,
    max_points: int,
    random_seed: int,
) -> tuple[Any, dict[str, Any]]:
    input_points = int(len(pcd.points))
    if max_points <= 0 or input_points <= max_points:
        return copy_pcd(pcd), {
            "input_points": input_points,
            "used_points": input_points,
            "target_points": int(max_points),
            "applied": False,
            "strategy": "identity",
            "voxel_size_m": 0.0,
            "random_subsample_applied": False,
        }

    pts = np.asarray(pcd.points, dtype=np.float64)
    extents = np.ptp(pts, axis=0) if len(pts) else np.zeros(3, dtype=np.float64)
    diag = float(np.linalg.norm(extents))
    base_voxel = max(diag / max(math.sqrt(float(max_points)), 1.0), 1e-4)
    scales = (0.45, 0.60, 0.75, 0.90, 1.0, 1.15, 1.35, 1.60, 2.0)
    best_pcd = None
    best_count = input_points
    best_voxel = 0.0
    best_key = (math.inf, 1, input_points)
    for scale in scales:
        voxel = float(base_voxel * scale)
        candidate = pcd.voxel_down_sample(voxel)
        count = int(len(candidate.points))
        if count <= 0:
            continue
        key = (abs(count - int(max_points)), 0 if count <= int(max_points) else 1, count)
        if key < best_key:
            best_key = key
            best_pcd = candidate
            best_count = count
            best_voxel = voxel
    if best_pcd is None:
        best_pcd = copy_pcd(pcd)
        best_count = int(len(best_pcd.points))
        best_voxel = 0.0

    random_subsample_applied = False
    if best_count > int(max_points):
        rng = np.random.default_rng(int(random_seed))
        keep = np.sort(rng.choice(best_count, int(max_points), replace=False)).tolist()
        best_pcd = best_pcd.select_by_index(keep)
        best_count = int(len(best_pcd.points))
        random_subsample_applied = True

    best_pcd = _reestimate_pcd_normals(best_pcd, cfg)
    return best_pcd, {
        "input_points": input_points,
        "used_points": best_count,
        "target_points": int(max_points),
        "applied": bool(best_count != input_points),
        "strategy": "adaptive_voxel_cap",
        "voxel_size_m": float(best_voxel),
        "random_subsample_applied": bool(random_subsample_applied),
        "bbox_diag_m": float(diag),
    }


def _load_dof_artifact_sample(artifact_dir: Path, is_replay: bool) -> dict[str, Any]:
    image_path = artifact_dir / "image_left.png"
    depth_path = artifact_dir / "dense_map_cam.npy"
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing image_left.png: {artifact_dir}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Missing dense_map_cam.npy: {artifact_dir}")

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = image_rgb.shape[:2]
    dense_raw = np.load(str(depth_path))
    depth = depth_to_meters(dense_raw, target_hw=(height, width))
    mask = _load_dof_mask(artifact_dir, height, width, is_replay=is_replay)
    intr = _load_dof_intrinsics(artifact_dir)
    request_data = _load_json_if_present(artifact_dir / "request.json")
    response_data = _load_json_if_present(artifact_dir / "response.json")
    extrinsics = _load_dof_extrinsics(artifact_dir, request_data)
    object_prompt = _load_dof_object_prompt(artifact_dir, is_replay=is_replay, request_data=request_data)
    experiment_config = _load_json_if_present(artifact_dir / "replay_experiment_config.json")
    cached_arrays = _load_cached_object_arrays(artifact_dir, is_replay=is_replay)
    object_obb = _load_reference_obb(artifact_dir, response_data, cached_arrays) if not is_replay else {}
    selected_replay_pose = _load_selected_replay_pose(artifact_dir) if is_replay else {}
    camera_intr = CameraIntrinsics(
        fx=intr["fx"],
        fy=intr["fy"],
        cx=intr["cx"],
        cy=intr["cy"],
        width=width,
        height=height,
    )
    return {
        "artifact_dir": artifact_dir,
        "rgb": image_rgb,
        "depth": depth,
        "dense_raw": np.asarray(dense_raw, dtype=np.float32),
        "mask": mask,
        "K": camera_intr,
        "request": request_data,
        "response": response_data,
        "extrinsics": extrinsics,
        "object_prompt": object_prompt,
        "experiment_config": experiment_config,
        "object_xyz": np.asarray(cached_arrays.get("point_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32),
        "object_rgb": np.asarray(cached_arrays.get("point_rgb", np.zeros((0, 3), dtype=np.float32)), dtype=np.float32),
        "object_pixels_xy": np.asarray(cached_arrays.get("pixels_xy", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32),
        "object_points": int(cached_arrays.get("points", 0)),
        "object_obb": object_obb,
        "selected_replay_pose": selected_replay_pose,
    }


def _draw_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = np.asarray(image_rgb, dtype=np.uint8).copy()
    mask_bool = np.asarray(mask) > 0
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]
    color_arr = np.asarray(color, dtype=np.float32)
    output[mask_bool] = np.clip(0.6 * output[mask_bool].astype(np.float32) + 0.4 * color_arr, 0, 255).astype(np.uint8)
    edge = mask_bool.astype(np.uint8) - cv2.erode(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    output[edge > 0] = color
    return output


def _project_points_to_image(points_cam: np.ndarray, intr: CameraIntrinsics, image_shape: tuple[int, int]) -> np.ndarray:
    points = np.asarray(points_cam, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    z = points[:, 2]
    valid = np.isfinite(points).all(axis=1) & (z > 1e-6)
    if not np.any(valid):
        return np.zeros((0, 2), dtype=np.int32)
    pts = points[valid]
    u = np.round((pts[:, 0] * intr.fx / pts[:, 2]) + intr.cx).astype(np.int32)
    v = np.round((pts[:, 1] * intr.fy / pts[:, 2]) + intr.cy).astype(np.int32)
    height, width = image_shape
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    if not np.any(inside):
        return np.zeros((0, 2), dtype=np.int32)
    return np.column_stack([u[inside], v[inside]]).astype(np.int32)


def _draw_registration_overlay(after_rgb: np.ndarray, after_mask: np.ndarray, aligned_pcd, intr: CameraIntrinsics) -> np.ndarray:
    left = _draw_mask_overlay(after_rgb, after_mask, color=(255, 166, 77))
    right = left.copy()
    uv = _project_points_to_image(np.asarray(aligned_pcd.points), intr, after_rgb.shape[:2])
    if len(uv) > 3000:
        keep = np.random.default_rng(42).choice(len(uv), 3000, replace=False)
        uv = uv[keep]
    proj_mask, uv_filtered = projected_uv_to_mask(uv, after_rgb.shape[:2], point_radius_px=1, connect_radius_px=3)
    for x, y in uv_filtered:
        cv2.circle(right, (int(x), int(y)), 1, (76, 175, 255), -1, cv2.LINE_AA)
    contours, _ = cv2.findContours(proj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(right, [largest], -1, (76, 175, 255), 2, cv2.LINE_AA)
    cv2.putText(left, "After target mask", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 210), 2, cv2.LINE_AA)
    cv2.putText(right, "After + aligned projection", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 210), 2, cv2.LINE_AA)
    divider = np.full((after_rgb.shape[0], 10, 3), 18, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def _resize_rgb_to_height(image_rgb: np.ndarray, target_height: int) -> np.ndarray:
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image must be non-empty.")
    if height == target_height:
        return image_rgb.copy()
    scale = float(target_height) / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _prepend_reference_panel_rgb(
    reference_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    gap_px: int = 8,
    background_rgb: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray:
    target_height = max(int(reference_rgb.shape[0]), int(overlay_rgb.shape[0]))
    panels = [
        _resize_rgb_to_height(reference_rgb, target_height),
        _resize_rgb_to_height(overlay_rgb, target_height),
    ]
    total_width = sum(int(panel.shape[1]) for panel in panels) + max(0, int(gap_px))
    canvas = np.full((target_height, total_width, 3), background_rgb, dtype=np.uint8)
    x = 0
    for index, panel in enumerate(panels):
        width = int(panel.shape[1])
        canvas[:, x:x + width] = panel
        x += width
        if index == 0:
            x += max(0, int(gap_px))
    return canvas


def _rigid_inverse(T: np.ndarray) -> np.ndarray:
    tform = np.asarray(T, dtype=np.float64).reshape(4, 4)
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = tform[:3, :3].T
    inv[:3, 3] = -(tform[:3, :3].T @ tform[:3, 3])
    return inv


def _transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    tform = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return (pts @ tform[:3, :3].T) + tform[:3, 3]


def _rotation_angle_deg(R: np.ndarray) -> float:
    rot = np.asarray(R, dtype=np.float64).reshape(3, 3)
    val = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))

def _projection_metrics(points_ref: np.ndarray, T_ref_to_cur: np.ndarray, cur_mask: np.ndarray, intr: CameraIntrinsics) -> dict[str, Any]:
    uv = _project_points_to_image(_transform_points(points_ref, T_ref_to_cur), intr, cur_mask.shape[:2])
    if len(uv) == 0:
        return {"projection_precision": 0.0, "projection_recall": 0.0, "projection_iou": 0.0, "projection_pixels": 0}
    pred_mask, _ = projected_uv_to_mask(uv, cur_mask.shape[:2], point_radius_px=1, connect_radius_px=3)
    pred = pred_mask > 0
    gt = np.asarray(cur_mask) > 0
    inter = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    return {
        "projection_precision": float(inter / max(pred_area, 1)),
        "projection_recall": float(inter / max(gt_area, 1)),
        "projection_iou": float(inter / max(union, 1)),
        "projection_pixels": int(pred_area),
    }


def _evaluate_registration_transform(
    src_pcd,
    tgt_pcd,
    T_candidate: np.ndarray,
    after_depth_m: np.ndarray,
    after_rgb: np.ndarray,
    after_mask: np.ndarray,
    after_intr: CameraIntrinsics,
    cfg: RegistrationConfig,
) -> dict[str, float]:
    visibility = visibility_aware_score(
        src_pcd_base=src_pcd,
        tgt_pcd_base=tgt_pcd,
        T_srcbase_to_tgtbase=T_candidate,
        wrist_depth=after_depth_m,
        wrist_rgb=after_rgb,
        wrist_intr=after_intr,
        T_wrist_to_base=np.eye(4, dtype=np.float64),
        cfg=cfg,
    )
    proj_metrics = _projection_metrics(np.asarray(src_pcd.points, dtype=np.float64), T_candidate, after_mask, after_intr)
    return {
        "visibility_score": float(visibility.get("score", 0.0)),
        "visible_count": float(visibility.get("visible_count", 0.0)),
        "depth_ok_ratio": float(visibility.get("depth_ok_ratio", 0.0)),
        "projection_iou": float(proj_metrics.get("projection_iou", 0.0)),
        "projection_precision": float(proj_metrics.get("projection_precision", 0.0)),
        "projection_recall": float(proj_metrics.get("projection_recall", 0.0)),
    }


def _colored_icp_guard_decision(
    T_gicp: np.ndarray,
    gicp_fitness: float,
    gicp_rmse: float,
    gicp_eval: dict[str, float],
    T_colored: np.ndarray,
    colored_fitness: float,
    colored_rmse: float,
    colored_eval: dict[str, float],
) -> dict[str, Any]:
    reasons: list[str] = []

    if not np.isfinite(colored_fitness) or colored_fitness <= 1e-6:
        reasons.append("colored_icp_no_valid_correspondence")
    if not np.isfinite(colored_rmse):
        reasons.append("colored_icp_invalid_rmse")

    if colored_eval.get("visibility_score", 0.0) <= -1e8 and gicp_eval.get("visibility_score", 0.0) > -1e8:
        reasons.append("colored_icp_lost_visibility")
    if colored_eval.get("projection_iou", 0.0) <= 1e-6 and gicp_eval.get("projection_iou", 0.0) >= 0.10:
        reasons.append("colored_icp_lost_projection")
    if gicp_fitness >= 0.20 and colored_fitness + 0.10 < gicp_fitness:
        reasons.append("colored_icp_fitness_drop")
    if (
        np.isfinite(gicp_rmse)
        and np.isfinite(colored_rmse)
        and gicp_rmse > 1e-6
        and colored_rmse > max(0.02, gicp_rmse * 1.8)
    ):
        reasons.append("colored_icp_rmse_spike")

    rot_delta = np.asarray(T_colored, dtype=np.float64)[:3, :3] @ np.asarray(T_gicp, dtype=np.float64)[:3, :3].T
    delta_angle_deg = _rotation_angle_deg(rot_delta)
    delta_trans_m = float(
        np.linalg.norm(
            np.asarray(T_colored, dtype=np.float64)[:3, 3] - np.asarray(T_gicp, dtype=np.float64)[:3, 3]
        )
    )
    severe_regression = (
        colored_eval.get("visibility_score", 0.0) <= -1e8 < gicp_eval.get("visibility_score", 0.0)
        or colored_eval.get("projection_iou", 0.0) + 0.10 < gicp_eval.get("projection_iou", 0.0)
        or (gicp_fitness >= 0.20 and colored_fitness + 0.10 < gicp_fitness)
    )
    if (delta_angle_deg > 35.0 or delta_trans_m > 0.05) and severe_regression:
        reasons.append("colored_icp_large_pose_jump")

    return {
        "reject": bool(reasons),
        "reasons": reasons,
        "delta_angle_deg": float(delta_angle_deg),
        "delta_trans_m": float(delta_trans_m),
        "gicp_visibility_score": float(gicp_eval.get("visibility_score", 0.0)),
        "colored_visibility_score": float(colored_eval.get("visibility_score", 0.0)),
        "gicp_projection_iou": float(gicp_eval.get("projection_iou", 0.0)),
        "colored_projection_iou": float(colored_eval.get("projection_iou", 0.0)),
    }


def _selection_score(reg_debug: dict[str, Any], proj_metrics: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    weight = 0.30
    target_rmse = 0.009
    target_projection_iou = 0.46
    target_projection_precision = 0.90
    target_projection_recall = 0.60
    target_reg_fitness = 0.955
    rmse = float(reg_debug.get("colored_icp_inlier_rmse", reg_debug.get("gicp_inlier_rmse", math.inf)))
    reg_fitness = float(reg_debug.get("colored_icp_fitness", reg_debug.get("gicp_fitness", 0.0)))
    projection_iou = float(proj_metrics.get("projection_iou", 0.0))
    projection_precision = float(proj_metrics.get("projection_precision", 0.0))
    projection_recall = float(proj_metrics.get("projection_recall", 0.0))
    rmse_term = min(1.5, rmse / max(target_rmse, 1e-6)) if np.isfinite(rmse) else 1.5
    # Precision alone can prefer a narrow but "clean" silhouette; recall keeps full-part coverage in the score.
    iou_penalty = 0.10 * max(0.0, target_projection_iou - projection_iou)
    precision_penalty = 0.14 * max(0.0, target_projection_precision - projection_precision)
    recall_penalty = 0.42 * max(0.0, target_projection_recall - projection_recall)
    fitness_penalty = 0.40 * max(0.0, target_reg_fitness - reg_fitness)
    final_penalty = iou_penalty + precision_penalty + recall_penalty + fitness_penalty
    invalid_penalty = 0.0
    if reg_fitness <= 1e-6:
        invalid_penalty += 0.8
    if projection_iou <= 1e-6:
        invalid_penalty += 0.8
    if projection_precision <= 1e-6:
        invalid_penalty += 0.4
    if projection_recall <= 1e-6:
        invalid_penalty += 0.4
    final_penalty += invalid_penalty
    score = (weight * rmse_term) + final_penalty
    return score, {
        "selection_weight": weight,
        "selection_target_rmse_m": target_rmse,
        "selection_target_projection_iou": target_projection_iou,
        "selection_target_projection_precision": target_projection_precision,
        "selection_target_projection_recall": target_projection_recall,
        "selection_target_reg_fitness": target_reg_fitness,
        "selection_reg_fitness": float(reg_fitness),
        "selection_iou_penalty": float(iou_penalty),
        "selection_precision_penalty": float(precision_penalty),
        "selection_recall_penalty": float(recall_penalty),
        "selection_fitness_penalty": float(fitness_penalty),
        "selection_invalid_penalty": float(invalid_penalty),
        "final_selection_penalty": float(final_penalty),
    }


def _prior_compare_keep_decision(
    best_result: dict[str, Any] | None,
    prior_result: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    debug = {
        "triggered": False,
        "best_tag": str(((best_result or {}).get("candidate") or SeedCandidate(tag="", T_init=np.eye(4))).tag),
        "prior_tag": str(((prior_result or {}).get("candidate") or SeedCandidate(tag="", T_init=np.eye(4))).tag),
        "best_projection_iou": 0.0,
        "best_projection_precision": 0.0,
        "best_projection_recall": 0.0,
        "best_rmse_m": math.inf,
        "best_fitness": 0.0,
        "prior_projection_iou": 0.0,
        "prior_projection_precision": 0.0,
        "prior_projection_recall": 0.0,
        "prior_rmse_m": math.inf,
        "prior_fitness": 0.0,
        "iou_gain": 0.0,
        "precision_gain": 0.0,
        "recall_gain": 0.0,
        "rmse_drop_m": 0.0,
        "fitness_gain": 0.0,
        "soft_tradeoff_accept": False,
        "reason": "",
    }
    if best_result is None or prior_result is None:
        return False, debug
    best_candidate = best_result.get("candidate")
    prior_candidate = prior_result.get("candidate")
    if not isinstance(best_candidate, SeedCandidate) or not isinstance(prior_candidate, SeedCandidate):
        return False, debug
    if str(best_candidate.tag) in {"tapir_prior_keep", "sam3d_prior_keep"}:
        return False, debug

    best_proj = dict(best_result.get("proj_metrics") or {})
    prior_proj = dict(prior_result.get("proj_metrics") or {})
    best_reg = dict(best_result.get("reg_debug") or {})
    prior_reg = dict(prior_result.get("reg_debug") or {})
    best_iou = float(best_proj.get("projection_iou", 0.0))
    prior_iou = float(prior_proj.get("projection_iou", 0.0))
    best_precision = float(best_proj.get("projection_precision", 0.0))
    prior_precision = float(prior_proj.get("projection_precision", 0.0))
    best_recall = float(best_proj.get("projection_recall", 0.0))
    prior_recall = float(prior_proj.get("projection_recall", 0.0))
    best_rmse = float(best_reg.get("colored_icp_inlier_rmse", best_reg.get("gicp_inlier_rmse", math.inf)))
    prior_rmse = float(prior_reg.get("colored_icp_inlier_rmse", prior_reg.get("gicp_inlier_rmse", math.inf)))
    best_fitness = float(best_reg.get("colored_icp_fitness", best_reg.get("gicp_fitness", 0.0)))
    prior_fitness = float(prior_reg.get("colored_icp_fitness", prior_reg.get("gicp_fitness", 0.0)))
    iou_gain = float(best_iou - prior_iou)
    precision_gain = float(best_precision - prior_precision)
    recall_gain = float(best_recall - prior_recall)
    rmse_drop = float(prior_rmse - best_rmse) if np.isfinite(prior_rmse) and np.isfinite(best_rmse) else 0.0
    fitness_gain = float(best_fitness - prior_fitness)
    improves_iou = bool(best_iou > prior_iou + 1e-4)
    improves_precision = bool(best_precision > prior_precision + 1e-4)
    improves_recall = bool(best_recall > prior_recall + 1e-4)
    improves_rmse = bool(best_rmse + 1e-6 < prior_rmse) if np.isfinite(prior_rmse) else bool(np.isfinite(best_rmse))
    improves_fitness = bool(best_fitness > prior_fitness + 1e-4)
    strong_accept = bool(
        improves_rmse
        and (
            (improves_iou and (improves_recall or improves_precision or improves_fitness))
            or (improves_recall and (improves_precision or improves_fitness))
            or (improves_precision and improves_fitness)
        )
    )
    soft_tradeoff_accept = bool(
        not strong_accept
        and improves_rmse
        and iou_gain >= -0.035
        and recall_gain >= 0.03
        and precision_gain >= -0.02
        and fitness_gain >= 0.04
        and rmse_drop >= max(0.0030, 0.20 * max(prior_rmse, 1e-6))
        and best_iou >= max(0.22, prior_iou - 0.035)
    )
    high_iou_prior_tradeoff_accept = bool(
        not strong_accept
        and improves_rmse
        and prior_iou >= 0.42
        and iou_gain >= -0.055
        and precision_gain >= -0.01
        and recall_gain >= -0.01
        and fitness_gain >= 0.06
        and rmse_drop >= max(0.0035, 0.22 * max(prior_rmse, 1e-6))
    )
    keep_prior = not (strong_accept or soft_tradeoff_accept or high_iou_prior_tradeoff_accept)
    reason = ""
    if keep_prior:
        missing = []
        if not improves_iou:
            missing.append("iou")
        if not improves_rmse:
            missing.append("rmse")
        if not improves_precision:
            missing.append("precision")
        if not improves_recall:
            missing.append("recall")
        if not improves_fitness:
            missing.append("fitness")
        reason = "refine_not_better_" + "_and_".join(missing)
    elif strong_accept:
        reason = "refine_multi_metric_win"
    elif soft_tradeoff_accept:
        reason = "refine_tradeoff_rmse_precision"
    else:
        reason = "refine_tradeoff_high_iou_prior"
    debug.update(
        {
            "triggered": bool(keep_prior),
            "best_projection_iou": float(best_iou),
            "best_projection_precision": float(best_precision),
            "best_projection_recall": float(best_recall),
            "best_rmse_m": float(best_rmse),
            "best_fitness": float(best_fitness),
            "prior_projection_iou": float(prior_iou),
            "prior_projection_precision": float(prior_precision),
            "prior_projection_recall": float(prior_recall),
            "prior_rmse_m": float(prior_rmse),
            "prior_fitness": float(prior_fitness),
            "iou_gain": float(iou_gain),
            "precision_gain": float(precision_gain),
            "recall_gain": float(recall_gain),
            "rmse_drop_m": float(rmse_drop),
            "fitness_gain": float(fitness_gain),
            "soft_tradeoff_accept": bool(soft_tradeoff_accept),
            "reason": reason,
        }
    )
    return keep_prior, debug


def _sample_projection_points(points: np.ndarray, max_points: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) <= max_points:
        return pts
    max_points = max(1, int(max_points))
    indices = np.linspace(0, len(pts) - 1, num=max_points, dtype=np.int32)
    return pts[indices]


def _prefilter_fallback_seed_candidates(
    seed_candidates: list[SeedCandidate],
    src_pcd,
    tgt_pcd,
    after: dict[str, Any],
    intr: CameraIntrinsics,
    cfg: RegistrationConfig,
    keep_top_k: int = 4,
) -> tuple[list[SeedCandidate], dict[str, Any]]:
    if len(seed_candidates) <= max(1, int(keep_top_k)):
        return list(seed_candidates), {
            "enabled": False,
            "reason": "candidate_count_not_exceed_topk",
            "keep_top_k": int(keep_top_k),
            "input_candidate_count": int(len(seed_candidates)),
            "selected_candidate_count": int(len(seed_candidates)),
            "selected_tags": [str(c.tag) for c in seed_candidates],
            "ranking": [],
        }

    ref_proj_points = _sample_projection_points(np.asarray(src_pcd.points, dtype=np.float64), max_points=2048)
    cheap_ranking: list[dict[str, Any]] = []
    for index, candidate in enumerate(seed_candidates):
        proj_metrics = _projection_metrics(
            ref_proj_points,
            np.asarray(candidate.T_init, dtype=np.float64).reshape(4, 4),
            after["mask"],
            intr,
        )
        proj_iou = float(proj_metrics.get("projection_iou", 0.0))
        proj_precision = float(proj_metrics.get("projection_precision", 0.0))
        proj_recall = float(proj_metrics.get("projection_recall", 0.0))
        cheap_score = (
            proj_iou
            + 0.28 * proj_precision
            + 0.40 * proj_recall
            + 0.05 * float(candidate.score_hint)
        )
        cheap_ranking.append(
            {
                "index": int(index),
                "tag": str(candidate.tag),
                "search_mode": str(candidate.search_mode or "full"),
                "cheap_score": float(cheap_score),
                "projection_iou": proj_iou,
                "projection_precision": proj_precision,
                "projection_recall": proj_recall,
                "score_hint": float(candidate.score_hint),
            }
        )

    cheap_sorted = sorted(
        cheap_ranking,
        key=lambda item: (
            -float(item["cheap_score"]),
            -float(item["projection_iou"]),
            -float(item["projection_recall"]),
            -float(item["projection_precision"]),
            int(item["index"]),
        ),
    )
    keep_n = max(1, min(int(keep_top_k), len(seed_candidates)))
    shortlist_n = max(keep_n + 1, min(len(seed_candidates), 6))
    shortlist_items = cheap_sorted[:shortlist_n]
    ranking: list[dict[str, Any]] = []
    for item in shortlist_items:
        candidate = seed_candidates[int(item["index"])]
        eval_metrics = _evaluate_registration_transform(
            src_pcd=src_pcd,
            tgt_pcd=tgt_pcd,
            T_candidate=np.asarray(candidate.T_init, dtype=np.float64).reshape(4, 4),
            after_depth_m=after["depth"],
            after_rgb=after["rgb"],
            after_mask=after["mask"],
            after_intr=intr,
            cfg=cfg,
        )
        vis = float(eval_metrics.get("visibility_score", 0.0))
        visible_count = float(eval_metrics.get("visible_count", 0.0))
        invalid = bool(
            vis <= -1e8
            or (
                item["projection_iou"] <= 1e-6
                and item["projection_precision"] <= 1e-6
                and item["projection_recall"] <= 1e-6
            )
        )
        pre_score = (
            float(item["cheap_score"])
            + 0.25 * float(np.clip(vis, 0.0, 1.0))
            + 0.02 * float(max(visible_count, 0.0) > 0.0)
        )
        ranking.append(
            {
                **item,
                "pre_score": float(pre_score),
                "invalid": bool(invalid),
                "visibility_score": vis,
                "visible_count": visible_count,
                "visibility_evaluated": True,
            }
        )
    shortlist_indices = {int(item["index"]) for item in shortlist_items}
    for item in cheap_sorted:
        if int(item["index"]) in shortlist_indices:
            continue
        ranking.append(
            {
                **item,
                "pre_score": float(item["cheap_score"]),
                "invalid": bool(
                    item["projection_iou"] <= 1e-6
                    and item["projection_precision"] <= 1e-6
                    and item["projection_recall"] <= 1e-6
                ),
                "visibility_score": float("nan"),
                "visible_count": 0.0,
                "visibility_evaluated": False,
            }
        )
    ranking_sorted = sorted(
        ranking,
        key=lambda item: (
            bool(item["invalid"]),
            not bool(item["visibility_evaluated"]),
            -float(item["pre_score"]),
            -float(item["projection_iou"]),
            -float(item["projection_recall"]),
            -float(item["projection_precision"]),
            int(item["index"]),
        ),
    )
    selected_items = [
        item
        for item in ranking_sorted
        if bool(item["visibility_evaluated"])
    ][:keep_n]
    selected_candidates: list[SeedCandidate] = []
    for item in selected_items:
        candidate = seed_candidates[int(item["index"])]
        selected_candidates.append(
            replace(
                candidate,
                metadata={
                    **(candidate.metadata or {}),
                    "prefilter_projection_iou": float(item["projection_iou"]),
                    "prefilter_projection_precision": float(item["projection_precision"]),
                    "prefilter_projection_recall": float(item["projection_recall"]),
                    "prefilter_visibility_score": float(item["visibility_score"]),
                    "prefilter_visible_count": float(item["visible_count"]),
                    "prefilter_cheap_score": float(item["cheap_score"]),
                    "prefilter_pre_score": float(item["pre_score"]),
                },
            )
        )
    return selected_candidates, {
        "enabled": True,
        "reason": "fallback_two_stage_prefilter",
        "keep_top_k": int(keep_n),
        "shortlist_count": int(shortlist_n),
        "input_candidate_count": int(len(seed_candidates)),
        "selected_candidate_count": int(len(selected_candidates)),
        "selected_tags": [str(candidate.tag) for candidate in selected_candidates],
        "ranking": ranking_sorted,
    }


def coarse_fine_registration(
    src_pcd,
    tgt_pcd,
    T_init: np.ndarray,
    after_depth_m: np.ndarray,
    after_rgb: np.ndarray,
    after_mask: np.ndarray,
    after_intr: CameraIntrinsics,
    cfg: RegistrationConfig,
    search_mode: str = "full",
) -> tuple[np.ndarray, dict[str, Any]]:
    debug: dict[str, Any] = {}
    t0 = time.perf_counter()
    T_camera = np.eye(4, dtype=np.float64)
    debug["search_mode"] = str(search_mode)
    refine_cfg = cfg

    if search_mode in {
        "tapir_local_fast",
        "tapir_local_verify",
        "tapir_micro_verify_strict",
        "tapir_micro_verify_relaxed",
        "sam3d_micro_verify_strict",
        "sam3d_micro_verify_relaxed",
        "fallback_light",
        "fallback_heavy",
    }:
        T_coarse = np.asarray(T_init, dtype=np.float64).reshape(4, 4)
        coarse_info = {"score": math.nan, "skipped": True}
        debug["coarse_score"] = float("nan")
        debug["coarse_ms"] = 0
        debug["coarse_skipped"] = True
    else:
        coarse_cfg = replace(
            cfg,
            search_rx_deg=(-45.0, -20.0, 0.0, 20.0, 45.0),
            search_ry_deg=(-45.0, -20.0, 0.0, 20.0, 45.0),
            search_rz_deg=(-120.0, -60.0, 0.0, 60.0, 120.0),
            search_tx_m=(-0.02, 0.0, 0.02),
            search_ty_m=(-0.02, 0.0, 0.02),
            search_tz_m=(-0.02, 0.0, 0.02),
        )
        T_coarse, coarse_info = local_hypothesis_search(
            src_head_pcd_base=src_pcd,
            tgt_pcd_base=tgt_pcd,
            T_init=T_init,
            wrist_depth_m=after_depth_m,
            wrist_rgb=after_rgb,
            wrist_intr=after_intr,
            T_wrist_to_base=T_camera,
            cfg=coarse_cfg,
        )
        debug["coarse_score"] = coarse_info.get("score", -1e9)
        debug["coarse_ms"] = int(round((time.perf_counter() - t0) * 1000.0))

    t1 = time.perf_counter()
    if search_mode in {"tapir_micro_verify_strict", "sam3d_micro_verify_strict"}:
        T_search = T_coarse
        fine_info = {"score": math.nan, "skipped": True}
        debug["fine_score"] = float("nan")
        debug["fine_ms"] = 0
        debug["fine_skipped"] = True
        refine_cfg = replace(
            cfg,
            gicp_max_iter=min(int(cfg.gicp_max_iter), 12),
            use_colored_icp=False,
            colored_icp_max_iter=0,
        )
    elif search_mode in {"tapir_micro_verify_relaxed", "sam3d_micro_verify_relaxed"}:
        T_search = T_coarse
        fine_info = {"score": math.nan, "skipped": True}
        debug["fine_score"] = float("nan")
        debug["fine_ms"] = 0
        debug["fine_skipped"] = True
        refine_cfg = replace(
            cfg,
            gicp_max_iter=min(int(cfg.gicp_max_iter), 20),
            use_colored_icp=False,
            colored_icp_max_iter=0,
        )
    elif search_mode == "tapir_local_fast":
        fine_cfg = replace(
            cfg,
            search_rx_deg=(-3.0, 0.0, 3.0),
            search_ry_deg=(-3.0, 0.0, 3.0),
            search_rz_deg=(-4.0, 0.0, 4.0),
            search_tx_m=(0.0,),
            search_ty_m=(0.0,),
            search_tz_m=(0.0,),
        )
        refine_cfg = replace(
            cfg,
            gicp_max_iter=min(int(cfg.gicp_max_iter), 36),
            colored_icp_max_iter=min(int(cfg.colored_icp_max_iter), 16),
        )
    elif search_mode == "tapir_local_verify":
        fine_cfg = replace(
            cfg,
            search_rx_deg=(-2.0, 0.0, 2.0),
            search_ry_deg=(-2.0, 0.0, 2.0),
            search_rz_deg=(-3.0, 0.0, 3.0),
            search_tx_m=(0.0,),
            search_ty_m=(0.0,),
            search_tz_m=(0.0,),
        )
        refine_cfg = replace(
            cfg,
            gicp_max_iter=min(int(cfg.gicp_max_iter), 18),
            colored_icp_max_iter=min(int(cfg.colored_icp_max_iter), 8),
        )
    elif search_mode == "fallback_light":
        fine_cfg = replace(
            cfg,
            search_rx_deg=(-6.0, 0.0, 6.0),
            search_ry_deg=(-6.0, 0.0, 6.0),
            search_rz_deg=(-10.0, 0.0, 10.0),
            search_tx_m=(0.0,),
            search_ty_m=(0.0,),
            search_tz_m=(0.0,),
        )
        refine_cfg = replace(
            cfg,
            gicp_max_iter=min(int(cfg.gicp_max_iter), 24),
            colored_icp_max_iter=min(int(cfg.colored_icp_max_iter), 12),
        )
    elif search_mode == "fallback_heavy":
        fine_cfg = replace(
            cfg,
            search_rx_deg=(-10.0, -5.0, 0.0, 5.0, 10.0),
            search_ry_deg=(-10.0, -5.0, 0.0, 5.0, 10.0),
            search_rz_deg=(-16.0, -8.0, 0.0, 8.0, 16.0),
            search_tx_m=(0.0,),
            search_ty_m=(0.0,),
            search_tz_m=(0.0,),
        )
        refine_cfg = replace(
            cfg,
            gicp_max_iter=min(int(cfg.gicp_max_iter), 42),
            colored_icp_max_iter=min(int(cfg.colored_icp_max_iter), 20),
        )
    else:
        fine_cfg = replace(
            cfg,
            search_rx_deg=(-8.0, -4.0, 0.0, 4.0, 8.0),
            search_ry_deg=(-8.0, -4.0, 0.0, 4.0, 8.0),
            search_rz_deg=(-10.0, -5.0, 0.0, 5.0, 10.0),
            search_tx_m=(-0.004, 0.0, 0.004),
            search_ty_m=(-0.004, 0.0, 0.004),
            search_tz_m=(-0.004, 0.0, 0.004),
        )
    if search_mode not in {
        "tapir_micro_verify_strict",
        "tapir_micro_verify_relaxed",
        "sam3d_micro_verify_strict",
        "sam3d_micro_verify_relaxed",
    }:
        T_search, fine_info = local_hypothesis_search(
            src_head_pcd_base=src_pcd,
            tgt_pcd_base=tgt_pcd,
            T_init=T_coarse,
            wrist_depth_m=after_depth_m,
            wrist_rgb=after_rgb,
            wrist_intr=after_intr,
            T_wrist_to_base=T_camera,
            cfg=fine_cfg,
        )
        debug["fine_score"] = fine_info.get("score", -1e9)
        debug["fine_ms"] = int(round((time.perf_counter() - t1) * 1000.0))

    t_gicp = time.perf_counter()
    T_gicp, gicp_res = refine_with_gicp(src_pcd, tgt_pcd, T_search, refine_cfg)
    debug["gicp_fitness"] = float(gicp_res.fitness)
    debug["gicp_inlier_rmse"] = float(gicp_res.inlier_rmse)
    debug["gicp_ms"] = int(round((time.perf_counter() - t_gicp) * 1000.0))
    gicp_eval = _evaluate_registration_transform(
        src_pcd=src_pcd,
        tgt_pcd=tgt_pcd,
        T_candidate=T_gicp,
        after_depth_m=after_depth_m,
        after_rgb=after_rgb,
        after_mask=after_mask,
        after_intr=after_intr,
        cfg=cfg,
    )
    debug["gicp_visibility_score"] = float(gicp_eval["visibility_score"])
    debug["gicp_projection_iou"] = float(gicp_eval["projection_iou"])

    T_final = T_gicp
    t_colored = time.perf_counter()
    if not bool(getattr(refine_cfg, "use_colored_icp", False)):
        debug["colored_icp_disabled"] = True
        debug["colored_icp_rejected"] = True
        debug["colored_icp_reject_reasons"] = ["colored_icp_disabled"]
        debug["colored_icp_effective_source"] = "gicp_fallback"
        debug["colored_icp_fitness"] = debug["gicp_fitness"]
        debug["colored_icp_inlier_rmse"] = debug["gicp_inlier_rmse"]
    else:
        try:
            T_colored, colored_res = refine_with_colored_icp(src_pcd, tgt_pcd, T_gicp, refine_cfg)
            colored_fitness = float(colored_res.fitness)
            colored_rmse = float(colored_res.inlier_rmse)
            colored_eval = _evaluate_registration_transform(
                src_pcd=src_pcd,
                tgt_pcd=tgt_pcd,
                T_candidate=T_colored,
                after_depth_m=after_depth_m,
                after_rgb=after_rgb,
                after_mask=after_mask,
                after_intr=after_intr,
                cfg=cfg,
            )
            guard = _colored_icp_guard_decision(
                T_gicp=T_gicp,
                gicp_fitness=debug["gicp_fitness"],
                gicp_rmse=debug["gicp_inlier_rmse"],
                gicp_eval=gicp_eval,
                T_colored=T_colored,
                colored_fitness=colored_fitness,
                colored_rmse=colored_rmse,
                colored_eval=colored_eval,
            )
            debug["colored_icp_raw_fitness"] = colored_fitness
            debug["colored_icp_raw_inlier_rmse"] = colored_rmse
            debug["colored_icp_raw_visibility_score"] = float(colored_eval["visibility_score"])
            debug["colored_icp_raw_projection_iou"] = float(colored_eval["projection_iou"])
            debug["colored_icp_guard"] = guard
            if guard["reject"]:
                debug["colored_icp_rejected"] = True
                debug["colored_icp_reject_reasons"] = list(guard["reasons"])
                debug["colored_icp_effective_source"] = "gicp_fallback"
                debug["colored_icp_fitness"] = debug["gicp_fitness"]
                debug["colored_icp_inlier_rmse"] = debug["gicp_inlier_rmse"]
            else:
                T_final = T_colored
                debug["colored_icp_rejected"] = False
                debug["colored_icp_reject_reasons"] = []
                debug["colored_icp_effective_source"] = "colored_icp"
                debug["colored_icp_fitness"] = colored_fitness
                debug["colored_icp_inlier_rmse"] = colored_rmse
        except Exception as exc:
            debug["colored_icp_error"] = str(exc).splitlines()[0]
            debug["colored_icp_rejected"] = True
            debug["colored_icp_reject_reasons"] = ["colored_icp_exception"]
            debug["colored_icp_effective_source"] = "gicp_fallback"
            debug["colored_icp_fitness"] = debug["gicp_fitness"]
            debug["colored_icp_inlier_rmse"] = debug["gicp_inlier_rmse"]
    debug["colored_icp_ms"] = int(round((time.perf_counter() - t_colored) * 1000.0))
    debug["total_reg_ms"] = int(round((time.perf_counter() - t0) * 1000.0))
    return T_final, debug


def _registration_eval_metrics(src_pcd, tgt_pcd, T_candidate: np.ndarray, cfg: RegistrationConfig) -> dict[str, float]:
    if o3d is None:
        return {"fitness": 0.0, "rmse_m": math.inf}
    try:
        eval_res = o3d.pipelines.registration.evaluate_registration(
            src_pcd,
            tgt_pcd,
            float(max(cfg.gicp_max_corr_dist, 1e-4)),
            np.asarray(T_candidate, dtype=np.float64).reshape(4, 4),
        )
        return {
            "fitness": float(getattr(eval_res, "fitness", 0.0)),
            "rmse_m": float(getattr(eval_res, "inlier_rmse", math.inf)),
        }
    except Exception:
        return {"fitness": 0.0, "rmse_m": math.inf}


def _maybe_override_fallback_light_result(
    candidate: SeedCandidate,
    T_candidate: np.ndarray,
    reg_debug: dict[str, Any],
    visibility: dict[str, Any],
    proj_metrics: dict[str, Any],
    src_pcd,
    tgt_pcd,
    after: dict[str, Any],
    cfg: RegistrationConfig,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any], dict[str, Any]]:
    if str(candidate.search_mode or "") not in {"fallback_light", "fallback_heavy"}:
        return T_candidate, reg_debug, visibility, proj_metrics
    metadata = candidate.metadata or {}
    raw_iou = float(metadata.get("prefilter_projection_iou", 0.0))
    raw_precision = float(metadata.get("prefilter_projection_precision", 0.0))
    raw_recall = float(metadata.get("prefilter_projection_recall", 0.0))
    if raw_iou <= 1e-6 and raw_precision <= 1e-6 and raw_recall <= 1e-6:
        return T_candidate, reg_debug, visibility, proj_metrics

    current_iou = float(proj_metrics.get("projection_iou", 0.0))
    current_precision = float(proj_metrics.get("projection_precision", 0.0))
    current_recall = float(proj_metrics.get("projection_recall", 0.0))
    iou_drop = raw_iou - current_iou
    precision_drop = raw_precision - current_precision
    recall_drop = raw_recall - current_recall
    current_rmse = float(reg_debug.get("colored_icp_inlier_rmse", reg_debug.get("gicp_inlier_rmse", math.inf)))
    collapse = bool(
        (raw_iou >= 0.45 and current_iou + 0.18 < raw_iou and current_precision + 0.12 < raw_precision)
        or (raw_recall >= 0.60 and current_recall + 0.12 < raw_recall and current_rmse >= 0.008)
        or (raw_iou >= 0.50 and current_iou < 0.25 and current_rmse >= 0.008)
    )
    if not collapse:
        return T_candidate, reg_debug, visibility, proj_metrics

    raw_T = np.asarray(candidate.T_init, dtype=np.float64).reshape(4, 4)
    raw_visibility = visibility_aware_score(
        src_pcd_base=src_pcd,
        tgt_pcd_base=tgt_pcd,
        T_srcbase_to_tgtbase=raw_T,
        wrist_depth=after["depth"],
        wrist_rgb=after["rgb"],
        wrist_intr=after["K"],
        T_wrist_to_base=np.eye(4, dtype=np.float64),
        cfg=cfg,
    )
    raw_proj_metrics = _projection_metrics(np.asarray(src_pcd.points, dtype=np.float64), raw_T, after["mask"], after["K"])
    raw_eval = _registration_eval_metrics(src_pcd, tgt_pcd, raw_T, cfg)
    updated_debug = dict(reg_debug)
    updated_debug["fallback_light_override"] = {
        "triggered": True,
        "reason": "prefilter_projection_regression",
        "raw_projection_iou": float(raw_iou),
        "raw_projection_precision": float(raw_precision),
        "raw_projection_recall": float(raw_recall),
        "refined_projection_iou": float(current_iou),
        "refined_projection_precision": float(current_precision),
        "refined_projection_recall": float(current_recall),
        "iou_drop": float(iou_drop),
        "precision_drop": float(precision_drop),
        "recall_drop": float(recall_drop),
    }
    updated_debug["search_mode"] = "fallback_prefilter_keep"
    updated_debug["colored_icp_effective_source"] = "fallback_prefilter_keep"
    updated_debug["gicp_fitness"] = float(raw_eval["fitness"])
    updated_debug["gicp_inlier_rmse"] = float(raw_eval["rmse_m"])
    updated_debug["colored_icp_fitness"] = float(raw_eval["fitness"])
    updated_debug["colored_icp_inlier_rmse"] = float(raw_eval["rmse_m"])
    updated_debug["gicp_visibility_score"] = float(raw_visibility.get("score", 0.0))
    updated_debug["gicp_projection_iou"] = float(raw_proj_metrics.get("projection_iou", 0.0))
    updated_debug["colored_icp_rejected"] = True
    updated_debug["colored_icp_reject_reasons"] = ["fallback_prefilter_keep"]
    return raw_T, updated_debug, raw_visibility, raw_proj_metrics


def assess_quality(
    reg_fitness: float,
    reg_rmse: float,
    visibility_score: float,
    projection_iou: float = 0.0,
    projection_precision: float = 0.0,
    projection_recall: float = 0.0,
) -> dict[str, Any]:
    quality = "good"
    reasons: list[str] = []
    strong_projection_support = bool(
        projection_iou >= 0.45
        and projection_precision >= 0.85
        and projection_recall >= 0.60
        and visibility_score >= 0.45
        and reg_fitness >= 0.95
    )
    if reg_fitness < 0.10:
        quality = "poor"
        reasons.append(f"low registration fitness ({reg_fitness:.3f})")
    elif reg_fitness < 0.35:
        quality = "fair"
        reasons.append(f"moderate registration fitness ({reg_fitness:.3f})")

    if not np.isfinite(reg_rmse) or reg_rmse > 0.030:
        quality = "poor"
        reasons.append(f"high registration RMSE ({reg_rmse:.4f}m)")
    elif reg_rmse > 0.010 and quality != "poor" and not strong_projection_support:
        quality = "fair"
        reasons.append(f"moderate registration RMSE ({reg_rmse:.4f}m)")

    if visibility_score < 0.05:
        quality = "poor"
        reasons.append(f"low visibility score ({visibility_score:.3f})")
    elif visibility_score < 0.18 and quality != "poor":
        quality = "fair"
        reasons.append(f"moderate visibility score ({visibility_score:.3f})")
    if projection_recall < 0.30:
        quality = "poor"
        reasons.append(f"low projection recall ({projection_recall:.3f})")
    elif projection_recall < 0.55 and quality != "poor":
        quality = "fair"
        reasons.append(f"moderate projection recall ({projection_recall:.3f})")
    return {"quality": quality, "reasons": reasons}


def run_reestimation(
    before: dict[str, Any],
    after: dict[str, Any],
    tapir_cfg: TapirMatchConfig | None = None,
    sam3d_cfg: Sam3dMatchConfig | None = None,
    cfg: RegistrationConfig | None = None,
    log_fn=None,
    init_method: str = "tapir",
) -> ReEstimationResult:
    if o3d is None:
        raise RuntimeError("open3d is required")
    cfg = cfg or RegistrationConfig()
    init_method_norm = str(init_method or "tapir").strip().lower()
    if init_method_norm not in {"tapir", "sam3d"}:
        raise ValueError(f"Unsupported init_method: {init_method}")

    t_total = time.perf_counter()
    debug: dict[str, Any] = {
        "mode": "camera",
        "init_candidate_mode": init_method_norm,
        "candidate_inits": [],
        "best_init": "",
        "tapir_init_enabled": bool(init_method_norm == "tapir"),
        "sam3d_init_enabled": bool(init_method_norm == "sam3d"),
        "init_method": init_method_norm,
    }
    stage_timings: dict[str, float] = {}

    t_stage = time.perf_counter()
    _emit_runtime_log(log_fn, "pointcloud/before start")
    before_pcd, before_downsample = _prepare_registration_pcd(before, cfg)
    stage_timings["pointcloud / before"] = time.perf_counter() - t_stage
    _emit_runtime_log(log_fn, f"pointcloud/before done {stage_timings['pointcloud / before'] * 1000.0:.1f} ms points={len(before_pcd.points)}")

    t_stage = time.perf_counter()
    _emit_runtime_log(log_fn, "pointcloud/after start")
    after_pcd, after_downsample = _prepare_registration_pcd(after, cfg)
    stage_timings["pointcloud / after"] = time.perf_counter() - t_stage
    _emit_runtime_log(log_fn, f"pointcloud/after done {stage_timings['pointcloud / after'] * 1000.0:.1f} ms points={len(after_pcd.points)}")

    debug["before_points"] = int(len(before_pcd.points))
    debug["after_points"] = int(len(after_pcd.points))
    debug["downsample_debug"] = {
        "reference": before_downsample,
        "current": after_downsample,
    }

    t_stage = time.perf_counter()
    init_result: Any
    init_debug: dict[str, Any]
    init_stage_prefix: str
    if init_method_norm == "tapir":
        if tapir_cfg is None:
            tapir_cfg = TapirMatchConfig()
        _emit_runtime_log(log_fn, "tapir/init start")
        init_result = estimate_tapir_init_transform(
            before_rgb=before["rgb"],
            before_depth_m=before["depth"],
            before_mask=before["mask"],
            after_rgb=after["rgb"],
            after_depth_m=after["depth"],
            after_mask=after["mask"],
            intr=before["K"],
            cfg=tapir_cfg,
        )
        debug["tapir_init"] = init_result.debug
        debug["_tapir_init_raw"] = init_result
        init_stage_prefix = "tapir"
        stage_timings["tapir / init"] = time.perf_counter() - t_stage
        init_debug = init_result.debug or {}
        tapir_timings = init_debug.get("timings_s") or {}
        track_e2e_s = float(tapir_timings.get("track", 0.0) or 0.0)
        track_remote_roundtrip_s = float(tapir_timings.get("track_remote_roundtrip", track_e2e_s) or track_e2e_s)
        stage_timings["tapir / init no_remote_roundtrip"] = max(
            0.0,
            float(stage_timings["tapir / init"]) - track_remote_roundtrip_s,
        )
        stage_timings["tapir / remote_roundtrip"] = max(
            0.0,
            float(stage_timings["tapir / init"]) - float(stage_timings["tapir / init no_remote_roundtrip"]),
        )
        track_no_rpc_io_s = float(tapir_timings.get("track_no_rpc_io", track_e2e_s) or track_e2e_s)
        stage_timings["tapir / init no_rpc_io"] = max(
            0.0,
            float(stage_timings["tapir / init"]) - track_e2e_s + track_no_rpc_io_s,
        )
        stage_timings["tapir / rpc_io_overhead"] = max(
            0.0,
            float(stage_timings["tapir / init"]) - float(stage_timings["tapir / init no_rpc_io"]),
        )
    else:
        if sam3d_cfg is None:
            sam3d_cfg = Sam3dMatchConfig()
        _emit_runtime_log(log_fn, "sam3d/init start")
        init_result = estimate_sam3d_init_transform(
            before=before,
            after=after,
            cfg=sam3d_cfg,
            log_fn=log_fn,
        )
        sam3d_prior_debug = evaluate_sam3d_prior(
            before=before,
            after=after,
            T_before_cam_to_after_cam=np.asarray(init_result.T_before_cam_to_after_cam, dtype=np.float64).reshape(4, 4),
            cfg=sam3d_cfg,
            reg_cfg=cfg,
            log_fn=log_fn,
        )
        init_result.debug.update({
            key: value for key, value in sam3d_prior_debug.items()
            if key != "timings_s"
        })
        init_result.debug["timings_s"] = {
            **(init_result.debug.get("timings_s") or {}),
            **(sam3d_prior_debug.get("timings_s") or {}),
        }
        init_result.debug["timings_s"]["total"] = float(time.perf_counter() - t_stage)
        init_result.debug["latency_ms"] = int(round(float(init_result.debug["timings_s"]["total"]) * 1000.0))
        before_pick_overlay = (
            np.asarray(init_result.before_pick_overlay_rgb, dtype=np.uint8)
            if init_result.before_pick_overlay_rgb is not None
            else np.zeros((0, 0, 3), dtype=np.uint8)
        )
        after_pick_overlay = (
            np.asarray(init_result.after_pick_overlay_rgb, dtype=np.uint8)
            if init_result.after_pick_overlay_rgb is not None
            else np.zeros((0, 0, 3), dtype=np.uint8)
        )
        debug["sam3d_init"] = init_result.debug
        debug["_sam3d_init_raw"] = init_result
        debug["_before_pick_overlay_rgb"] = before_pick_overlay
        debug["_after_pick_overlay_rgb"] = after_pick_overlay
        init_stage_prefix = "sam3d"
        stage_timings["sam3d / init"] = time.perf_counter() - t_stage
        init_debug = init_result.debug or {}
        sam3d_remote_roundtrip_s = float((init_debug.get("timings_s") or {}).get("remote_roundtrip", 0.0) or 0.0)
        stage_timings["sam3d / init no_remote_roundtrip"] = max(
            0.0,
            float(stage_timings["sam3d / init"]) - sam3d_remote_roundtrip_s,
        )
        stage_timings["sam3d / remote_roundtrip"] = max(
            0.0,
            float(stage_timings["sam3d / init"]) - float(stage_timings["sam3d / init no_remote_roundtrip"]),
        )
        stage_timings["sam3d / infer"] = float((init_debug.get("timings_s") or {}).get("infer", 0.0) or 0.0)
        stage_timings["sam3d / prior_eval"] = float((init_debug.get("timings_s") or {}).get("prior_eval", 0.0) or 0.0)
        stage_timings["sam3d / completion_registration"] = float(
            (init_debug.get("timings_s") or {}).get("completion_registration", 0.0) or 0.0
        )
    _emit_runtime_log(
        log_fn,
        f"{init_stage_prefix}/init done {stage_timings[f'{init_stage_prefix} / init'] * 1000.0:.1f} ms "
        f"success={bool(getattr(init_result, 'success', False))} "
        f"valid3d={(init_debug or {}).get('valid_3d_pairs', 0)} "
        f"inliers={(init_debug or {}).get('inlier_count', (init_debug or {}).get('num_inliers', 0))}",
    )

    t_reg_prep = time.perf_counter()
    if init_method_norm == "sam3d":
        seed_candidates, seed_debug = _build_sam3d_seed_candidates(before, after, init_result, init_debug)
    else:
        seed_candidates, seed_debug = _build_seed_candidates(before, after, init_result, init_debug)
    if not seed_candidates:
        reason = init_debug.get("reason") or init_debug.get("error") or "no valid init seed"
        raise RuntimeError(f"{init_method_norm.upper()} init failed: {reason}")
    debug["candidate_inits"] = [cand.tag for cand in seed_candidates]
    debug["candidate_search_modes"] = [cand.search_mode for cand in seed_candidates]
    debug["seed_policy"] = seed_debug.get("seed_policy")
    _emit_runtime_log(
        log_fn,
        f"seed candidates prepared count={len(seed_candidates)} policy={seed_debug.get('seed_policy', '')} "
        f"tags={debug['candidate_inits'][:8]}",
    )

    registration_before_pcd = before_pcd
    registration_after_pcd = after_pcd
    working_clouds_debug = {
        "reference": {
            "input_points": int(len(before_pcd.points)),
            "used_points": int(len(before_pcd.points)),
            "target_points": 0,
            "applied": False,
            "strategy": "identity",
            "voxel_size_m": 0.0,
            "random_subsample_applied": False,
        },
        "current": {
            "input_points": int(len(after_pcd.points)),
            "used_points": int(len(after_pcd.points)),
            "target_points": 0,
            "applied": False,
            "strategy": "identity",
            "voxel_size_m": 0.0,
            "random_subsample_applied": False,
        },
    }
    seed_policy = str(seed_debug.get("seed_policy") or "")
    working_target_points = _registration_working_target_points(seed_policy)
    if working_target_points > 0:
        registration_before_pcd, working_clouds_debug["reference"] = _cap_registration_working_pcd(
            before_pcd,
            cfg=cfg,
            max_points=int(working_target_points),
            random_seed=17,
        )
        registration_after_pcd, working_clouds_debug["current"] = _cap_registration_working_pcd(
            after_pcd,
            cfg=cfg,
            max_points=int(working_target_points),
            random_seed=29,
        )
        _emit_runtime_log(
            log_fn,
            "registration working clouds "
            f"ref={len(registration_before_pcd.points)}/{len(before_pcd.points)} "
            f"cur={len(registration_after_pcd.points)}/{len(after_pcd.points)} "
            f"target={working_target_points}",
        )
    debug["working_clouds_debug"] = working_clouds_debug
    if seed_policy == "fallback_search":
        seed_candidates, fallback_prefilter_debug = _prefilter_fallback_seed_candidates(
            seed_candidates,
            src_pcd=registration_before_pcd,
            tgt_pcd=registration_after_pcd,
            after=after,
            intr=before["K"],
            cfg=cfg,
            keep_top_k=5,
        )
        seed_debug["fallback_prefilter"] = fallback_prefilter_debug
        if bool(fallback_prefilter_debug.get("enabled")):
            debug["candidate_inits"] = [cand.tag for cand in seed_candidates]
            debug["candidate_search_modes"] = [cand.search_mode for cand in seed_candidates]
            _emit_runtime_log(
                log_fn,
                "fallback prefilter "
                f"kept={fallback_prefilter_debug.get('selected_candidate_count', 0)}/"
                f"{fallback_prefilter_debug.get('input_candidate_count', 0)} "
                f"tags={fallback_prefilter_debug.get('selected_tags', [])[:8]}",
            )

    stage_timings["registration / prep"] = time.perf_counter() - t_reg_prep
    _emit_runtime_log(log_fn, f"registration/prep done {stage_timings['registration / prep'] * 1000.0:.1f} ms")

    best_result: dict[str, Any] | None = None
    candidate_ranking: list[dict[str, Any]] = []
    t_stage = time.perf_counter()
    ref_proj_points = np.asarray(before_pcd.points, dtype=np.float64)
    prior_result: dict[str, Any] | None = None
    index = 0
    while index < len(seed_candidates):
        candidate = seed_candidates[index]
        t_candidate = time.perf_counter()
        _emit_runtime_log(log_fn, f"seed[{index}] start tag={candidate.tag}")
        if str(candidate.search_mode or "") in {"tapir_prior_keep", "sam3d_prior_keep"}:
            T_candidate = np.asarray(candidate.T_init, dtype=np.float64).reshape(4, 4)
            reg_debug = _prior_keep_debug(candidate)
        else:
            T_candidate, reg_debug = coarse_fine_registration(
                src_pcd=registration_before_pcd,
                tgt_pcd=registration_after_pcd,
                T_init=np.asarray(candidate.T_init, dtype=np.float64).reshape(4, 4),
                after_depth_m=after["depth"],
                after_rgb=after["rgb"],
                after_mask=after["mask"],
                after_intr=before["K"],
                cfg=cfg,
                search_mode=str(candidate.search_mode or "full"),
        )
        visibility = visibility_aware_score(
            src_pcd_base=registration_before_pcd,
            tgt_pcd_base=registration_after_pcd,
            T_srcbase_to_tgtbase=T_candidate,
            wrist_depth=after["depth"],
            wrist_rgb=after["rgb"],
            wrist_intr=before["K"],
            T_wrist_to_base=np.eye(4, dtype=np.float64),
            cfg=cfg,
        )
        proj_metrics = _projection_metrics(ref_proj_points, T_candidate, after["mask"], before["K"])
        T_candidate, reg_debug, visibility, proj_metrics = _maybe_override_fallback_light_result(
            candidate=candidate,
            T_candidate=T_candidate,
            reg_debug=reg_debug,
            visibility=visibility,
            proj_metrics=proj_metrics,
            src_pcd=registration_before_pcd,
            tgt_pcd=registration_after_pcd,
            after=after,
            cfg=cfg,
        )
        selection_score_raw, selection_meta = _selection_score(reg_debug, proj_metrics)
        selection_bias = 0.05 * float(candidate.score_hint)
        selection_score = float(selection_score_raw + selection_bias)
        candidate_info = {
            "index": index,
            "tag": candidate.tag,
            "selection_score": float(selection_score),
            "selection_score_raw": float(selection_score_raw),
            "selection_bias": float(selection_bias),
            "visibility_score": float(visibility.get("score", 0.0)),
            "projection_iou": float(proj_metrics.get("projection_iou", 0.0)),
            "projection_precision": float(proj_metrics.get("projection_precision", 0.0)),
            "projection_recall": float(proj_metrics.get("projection_recall", 0.0)),
            "rmse_m": float(reg_debug.get("colored_icp_inlier_rmse", reg_debug.get("gicp_inlier_rmse", math.inf))),
            "gicp_fitness": float(reg_debug.get("gicp_fitness", 0.0)),
            "metadata": candidate.metadata,
            "search_mode": str(candidate.search_mode or "full"),
        }
        candidate_ranking.append(candidate_info)
        _emit_runtime_log(
            log_fn,
            f"seed[{index}] done tag={candidate.tag} total={(time.perf_counter() - t_candidate) * 1000.0:.1f} ms "
            f"score={selection_score:.4f} rmse={candidate_info['rmse_m']:.4f} "
            f"proj_iou={candidate_info['projection_iou']:.3f} "
            f"recall={candidate_info['projection_recall']:.3f} "
            f"vis={candidate_info['visibility_score']:.3f}",
        )
        if str(candidate.tag) in {"tapir_prior_keep", "sam3d_prior_keep"}:
            prior_result = {
                "candidate": candidate,
                "candidate_index": index,
                "T_registration": T_candidate,
                "reg_debug": dict(reg_debug),
                "visibility": dict(visibility),
                "proj_metrics": dict(proj_metrics),
                "selection_score": float(selection_score),
                "selection_meta": dict(selection_meta),
            }
        score_key = (
            float(selection_score),
            -float(proj_metrics.get("projection_recall", 0.0)),
            -float(visibility.get("score", 0.0)),
            float(reg_debug.get("colored_icp_inlier_rmse", reg_debug.get("gicp_inlier_rmse", math.inf))),
            -float(proj_metrics.get("projection_precision", 0.0)),
            -float(proj_metrics.get("projection_iou", 0.0)),
        )
        if best_result is None or score_key < best_result["score_key"]:
            best_result = {
                "candidate": candidate,
                "candidate_index": index,
                "T_registration": T_candidate,
                "reg_debug": reg_debug,
                "visibility": visibility,
                "proj_metrics": proj_metrics,
                "selection_score": float(selection_score),
                "selection_meta": selection_meta,
                "score_key": score_key,
            }
        index += 1
    if seed_policy in {"tapir_prior_compare", "sam3d_prior_compare"}:
        keep_prior, prior_compare_debug = _prior_compare_keep_decision(best_result, prior_result)
        seed_debug["prior_compare"] = prior_compare_debug
        if keep_prior and prior_result is not None:
            best_result = prior_result
            _emit_runtime_log(
                log_fn,
                "prior-compare keep prior "
                f"reason={prior_compare_debug.get('reason', '')} "
                f"iou_gain={float(prior_compare_debug.get('iou_gain', 0.0)):.3f} "
                f"rmse_drop={float(prior_compare_debug.get('rmse_drop_m', 0.0)):.4f}",
            )
    stage_timings["registration / refine"] = time.perf_counter() - t_stage
    stage_timings["registration / total"] = (
        float(stage_timings.get("registration / prep", 0.0))
        + float(stage_timings["registration / refine"])
    )
    _emit_runtime_log(log_fn, f"registration/refine done {stage_timings['registration / refine'] * 1000.0:.1f} ms")
    _emit_runtime_log(log_fn, f"registration/total done {stage_timings['registration / total'] * 1000.0:.1f} ms")

    assert best_result is not None
    T_registration = np.asarray(best_result["T_registration"], dtype=np.float64).reshape(4, 4)
    reg_debug = dict(best_result["reg_debug"])
    visibility = dict(best_result["visibility"])
    proj_metrics = dict(best_result["proj_metrics"])
    debug.update({f"reg_{key}": value for key, value in reg_debug.items()})
    debug["reg_visibility_score"] = float(visibility.get("score", 0.0))
    debug["reg_visible_count"] = int(round(float(visibility.get("visible_count", 0.0))))
    debug["reg_depth_ok_ratio"] = float(visibility.get("depth_ok_ratio", 0.0))
    debug["reg_projection_precision"] = float(proj_metrics.get("projection_precision", 0.0))
    debug["reg_projection_recall"] = float(proj_metrics.get("projection_recall", 0.0))
    debug["reg_projection_iou"] = float(proj_metrics.get("projection_iou", 0.0))
    debug["best_init"] = str(best_result["candidate"].tag)
    debug["best_score"] = float(best_result["selection_score"])
    debug["candidate_ranking"] = sorted(candidate_ranking, key=lambda item: item["selection_score"])[:5]
    debug["selection_score"] = float(best_result["selection_score"])
    debug["selection_meta"] = dict(best_result["selection_meta"])
    debug["registration_summary"] = {
        "transform_ref_to_cur": T_registration.tolist(),
        "transform_cur_to_ref": _rigid_inverse(T_registration).tolist(),
        "fitness": float(reg_debug.get("colored_icp_fitness", reg_debug.get("gicp_fitness", 0.0))),
        "rmse_m": float(reg_debug.get("colored_icp_inlier_rmse", reg_debug.get("gicp_inlier_rmse", math.inf))),
        "coarse_score": float(reg_debug.get("coarse_score", -1e9)),
        "candidate_index": (
            -2
            if best_result["candidate"].tag in {"tapir_prior_local_refine", "sam3d_prior_local_refine"}
            else int(best_result["candidate_index"])
        ),
        "candidate_count": int(len(seed_candidates)),
        "rotation_change_deg": _rotation_angle_deg(T_registration[:3, :3]),
        "projection_precision": float(proj_metrics.get("projection_precision", 0.0)),
        "projection_recall": float(proj_metrics.get("projection_recall", 0.0)),
        "projection_iou": float(proj_metrics.get("projection_iou", 0.0)),
        "search_mode": str(reg_debug.get("search_mode", "")),
        "coarse_ms": int(reg_debug.get("coarse_ms", 0) or 0),
        "fine_ms": int(reg_debug.get("fine_ms", 0) or 0),
        "gicp_ms": int(reg_debug.get("gicp_ms", 0) or 0),
        "colored_icp_ms": int(reg_debug.get("colored_icp_ms", 0) or 0),
        "total_reg_ms": int(reg_debug.get("total_reg_ms", 0) or 0),
        "colored_icp_effective_source": str(reg_debug.get("colored_icp_effective_source", "")),
        "selection_score": float(best_result["selection_score"]),
        "surface_reference_points": int(len(registration_before_pcd.points)),
        "surface_current_points": int(len(registration_after_pcd.points)),
        "seed_debug": {
            **seed_debug,
            "selected_candidate_tag": str(best_result["candidate"].tag),
            "selected_candidate_augmented": bool(best_result["candidate"].metadata.get("extra_angle_deg") is not None),
            "selected_candidate_extra_angle_deg": best_result["candidate"].metadata.get("extra_angle_deg"),
            "chosen_seed_index": int(best_result["candidate_index"]),
            "chosen_seed_tag": str(best_result["candidate"].tag),
            "chosen_seed_augmented": bool(best_result["candidate"].metadata.get("extra_angle_deg") is not None),
            "chosen_seed_extra_angle_deg": best_result["candidate"].metadata.get("extra_angle_deg"),
            "candidate_ranking_topk": sorted(candidate_ranking, key=lambda item: item["selection_score"])[:5],
            "prior_debug": {
                **(init_debug or {}),
                **best_result["selection_meta"],
            },
            f"{init_method_norm}_prior": {
                **(init_debug or {}),
                **best_result["selection_meta"],
            },
        },
        "downsample_debug": {
            "reference": before_downsample,
            "current": after_downsample,
        },
        "working_clouds_debug": working_clouds_debug,
    }

    quality = assess_quality(
        float(reg_debug.get("colored_icp_fitness", reg_debug.get("gicp_fitness", 0.0))),
        float(reg_debug.get("colored_icp_inlier_rmse", reg_debug.get("gicp_inlier_rmse", math.inf))),
        float(visibility.get("score", 0.0)),
        float(proj_metrics.get("projection_iou", 0.0)),
        float(proj_metrics.get("projection_precision", 0.0)),
        float(proj_metrics.get("projection_recall", 0.0)),
    )
    debug["quality"] = quality
    debug["stage_timings_s"] = stage_timings
    elapsed_s = time.perf_counter() - t_total
    debug["total_elapsed_s"] = elapsed_s
    _emit_runtime_log(
        log_fn,
        f"run done total={elapsed_s * 1000.0:.1f} ms best_init={debug['best_init']} "
        f"quality={quality['quality']} reg_rmse={debug.get('reg_colored_icp_inlier_rmse', debug.get('reg_gicp_inlier_rmse'))}",
    )

    return ReEstimationResult(
        success=quality["quality"] != "poor",
        mode="camera",
        T_registration=T_registration,
        T_slip=T_registration.copy(),
        debug=debug,
        elapsed_s=elapsed_s,
    )


def _parse_replay_dir_name(replay_dir_name: str) -> str:
    parts = replay_dir_name.split("_")
    if len(parts) >= 2:
        return parts[1][:6]
    return replay_dir_name[:20]


def _result_row(pair: dict[str, Any], save_dir: Path | None = None, init_method: str = "tapir") -> dict[str, Any]:
    before_dir = pair.get("before_dir")
    replay_dir = pair.get("replay_dir")
    log_root = pair.get("log_root")
    return {
        "case_name": _parse_replay_dir_name(replay_dir.name if isinstance(replay_dir, Path) else ""),
        "ok": False,
        "error": "",
        "log_root": str(log_root) if log_root else "",
        "before_dir": str(before_dir) if before_dir else "",
        "replay_dir": str(replay_dir) if replay_dir else "",
        "memory_dir": pair.get("memory_dir", ""),
        "before_id": before_dir.name if isinstance(before_dir, Path) else "",
        "replay_id": replay_dir.name if isinstance(replay_dir, Path) else "",
        "init_method": str(init_method or "tapir"),
        "save_dir": str(save_dir) if save_dir else "",
    }


def _case_output_dirname(pair: dict[str, Any], seen_names: set[str] | None = None) -> str:
    replay_dir = pair.get("replay_dir")
    log_root = pair.get("log_root")
    if not isinstance(replay_dir, Path):
        return "unknown_case"
    base_name = _parse_replay_dir_name(replay_dir.name)
    if seen_names is None:
        return base_name
    if base_name not in seen_names:
        seen_names.add(base_name)
        return base_name
    log_name = log_root.name if isinstance(log_root, Path) else "log"
    candidate = f"{log_name}__{base_name}"
    suffix = 2
    while candidate in seen_names:
        candidate = f"{log_name}__{base_name}__{suffix}"
        suffix += 1
    seen_names.add(candidate)
    return candidate


def _fill_result_row(row: dict[str, Any], result: ReEstimationResult) -> None:
    debug = result.debug or {}
    init_method = str(debug.get("init_method") or row.get("init_method") or "tapir")
    init_debug = (
        debug.get("tapir_init", {}) if init_method == "tapir" else debug.get("sam3d_init", {})
    ) or {}
    quality = debug.get("quality", {}) or {}
    row.update(
        {
            "ok": True,
            "quality": quality.get("quality", ""),
            "elapsed_s": result.elapsed_s,
            "best_init": debug.get("best_init", "tapir"),
            "best_score": debug.get("best_score"),
            "init_method": init_method,
            "init_success": init_debug.get("success"),
            "init_backend": init_debug.get("backend"),
            "init_api_url": init_debug.get("api_url"),
            "init_matches": init_debug.get("num_matches"),
            "init_valid3d": init_debug.get("num_valid_3d_matches"),
            "init_inliers": init_debug.get("num_inliers"),
            "init_inlier_ratio": init_debug.get("inlier_ratio"),
            "init_rmse_m": init_debug.get("ransac_rmse_m", init_debug.get("rmse_m")),
            "init_e2e_ms": init_debug.get("tapir_init_latency_ms", init_debug.get("latency_ms")),
            "init_no_remote_roundtrip_ms": init_debug.get(
                "tapir_init_latency_no_remote_roundtrip_ms",
                init_debug.get("latency_ms"),
            ),
            "init_no_rpc_io_ms": init_debug.get(
                "tapir_init_latency_no_rpc_io_ms",
                init_debug.get("latency_ms"),
            ),
            "tapir_success": init_debug.get("success") if init_method == "tapir" else "",
            "tapir_backend": init_debug.get("backend") if init_method == "tapir" else "",
            "tapir_remote_host": init_debug.get("remote_host") if init_method == "tapir" else "",
            "tapir_remote_port": init_debug.get("remote_port") if init_method == "tapir" else "",
            "tapir_matches": init_debug.get("num_matches") if init_method == "tapir" else "",
            "tapir_valid3d": init_debug.get("num_valid_3d_matches") if init_method == "tapir" else "",
            "tapir_inliers": init_debug.get("num_inliers") if init_method == "tapir" else "",
            "tapir_inlier_ratio": init_debug.get("inlier_ratio") if init_method == "tapir" else "",
            "tapir_rmse_m": init_debug.get("ransac_rmse_m") if init_method == "tapir" else "",
            "tapir_track_e2e_ms": init_debug.get("tapir_latency_ms") if init_method == "tapir" else "",
            "tapir_track_no_remote_roundtrip_ms": init_debug.get("tapir_latency_no_remote_roundtrip_ms") if init_method == "tapir" else "",
            "tapir_track_no_rpc_io_ms": init_debug.get("tapir_latency_no_rpc_io_ms") if init_method == "tapir" else "",
            "tapir_init_e2e_ms": init_debug.get("tapir_init_latency_ms") if init_method == "tapir" else "",
            "tapir_init_no_remote_roundtrip_ms": init_debug.get("tapir_init_latency_no_remote_roundtrip_ms") if init_method == "tapir" else "",
            "tapir_init_no_rpc_io_ms": init_debug.get("tapir_init_latency_no_rpc_io_ms") if init_method == "tapir" else "",
            "visibility_score": debug.get("reg_visibility_score"),
            "visible_count": debug.get("reg_visible_count"),
            "depth_ok_ratio": debug.get("reg_depth_ok_ratio"),
            "gicp_fitness": debug.get("reg_gicp_fitness"),
            "gicp_rmse_m": debug.get("reg_gicp_inlier_rmse"),
            "colored_icp_fitness": debug.get("reg_colored_icp_fitness"),
            "colored_icp_rmse_m": debug.get("reg_colored_icp_inlier_rmse"),
        }
    )


def _write_pair_outputs(
    save_dir: Path,
    pair: dict[str, Any],
    before: dict[str, Any],
    after: dict[str, Any],
    result: ReEstimationResult,
    cfg_used: RegistrationConfig,
) -> None:
    init_method = str((result.debug or {}).get("init_method") or "tapir")
    init_stage_prefix = "tapir" if init_method == "tapir" else "sam3d"
    init_debug_payload = dict((result.debug or {}).get(f"{init_method}_init") or {})
    save_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "error.txt",
        "tapir_error.txt",
        "sam3d_error.txt",
        "tapir.ply",
        "tapir_overlay.png",
        "tapir_matches_overlay.png",
        "sam3d_completion.ply",
        "sam3d_before_pick_overlay.png",
        "sam3d_after_pick_overlay.png",
        "left_replay_object_overlay.png",
    ):
        stale_path = save_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    for stale_pattern in (
        "left_replay_object_overlay_no_trim_tapir_*.png",
        "tapir_correspondence_overlay_*.png",
        "left_replay_object_tapir_prior_overlay_*.png",
        "left_replay_object_sam3d_prior_overlay*.png",
        "left_replay_triptych_*.png",
    ):
        for stale_path in save_dir.glob(stale_pattern):
            if stale_path.is_file():
                stale_path.unlink()

    bp, before_downsample = _prepare_registration_pcd(before, cfg_used)
    ap, after_downsample = _prepare_registration_pcd(after, cfg_used)
    aligned = copy_pcd(bp)
    aligned.transform(result.T_registration)

    o3d.io.write_point_cloud(str(save_dir / "before.ply"), bp)
    o3d.io.write_point_cloud(str(save_dir / "after.ply"), ap)
    o3d.io.write_point_cloud(str(save_dir / "reference_object_transformed_colored.ply"), aligned)

    overlay = _draw_registration_overlay(after["rgb"], after["mask"], aligned, after["K"])
    overlay_with_reference = _prepend_reference_panel_rgb(before["rgb"], overlay)
    cv2.imwrite(str(save_dir / "left_replay_overlay.png"), cv2.cvtColor(overlay_with_reference, cv2.COLOR_RGB2BGR))

    init_raw = result.debug.get("_tapir_init_raw") if init_method == "tapir" else result.debug.get("_sam3d_init_raw")
    if init_method == "tapir" and init_raw is not None:
        tapir_match_vis = draw_tapir_matches(before["rgb"], after["rgb"], init_raw)
        requested_points = int(((result.debug.get("tapir_init") or {}).get("requested_points")) or 0)
        mode = str(((result.debug.get("tapir_init") or {}).get("sampling_mode")) or "uniform_random")
        if requested_points > 0:
            cv2.imwrite(
                str(save_dir / f"tapir_correspondence_overlay_{mode}_{requested_points}.png"),
                cv2.cvtColor(tapir_match_vis, cv2.COLOR_RGB2BGR),
            )
        init_T = _reconstruct_tapir_transform(init_raw)
        if init_T is not None:
            tapir_aligned = copy_pcd(bp)
            tapir_aligned.transform(init_T)
            tapir_overlay = _draw_registration_overlay(after["rgb"], after["mask"], tapir_aligned, after["K"])
            if requested_points > 0:
                tapir_overlay_with_reference = _prepend_reference_panel_rgb(before["rgb"], tapir_overlay)
                cv2.imwrite(
                    str(save_dir / f"left_replay_object_tapir_prior_overlay_{mode}_{requested_points}.png"),
                    cv2.cvtColor(tapir_overlay_with_reference, cv2.COLOR_RGB2BGR),
                )
    elif init_method == "sam3d":
        before_pick_overlay = np.asarray(result.debug.get("_before_pick_overlay_rgb", np.zeros((0, 0, 3), dtype=np.uint8)), dtype=np.uint8)
        after_pick_overlay = np.asarray(result.debug.get("_after_pick_overlay_rgb", np.zeros((0, 0, 3), dtype=np.uint8)), dtype=np.uint8)
        if before_pick_overlay.size > 0:
            cv2.imwrite(
                str(save_dir / "sam3d_before_pick_overlay.png"),
                cv2.cvtColor(before_pick_overlay, cv2.COLOR_RGB2BGR),
            )
        if after_pick_overlay.size > 0:
            cv2.imwrite(
                str(save_dir / "sam3d_after_pick_overlay.png"),
                cv2.cvtColor(after_pick_overlay, cv2.COLOR_RGB2BGR),
            )
        init_T = _reconstruct_tapir_transform(init_raw)
        if init_T is not None:
            sam3d_aligned = copy_pcd(bp)
            sam3d_aligned.transform(init_T)
            sam3d_overlay = _draw_registration_overlay(after["rgb"], after["mask"], sam3d_aligned, after["K"])
            sam3d_overlay_with_reference = _prepend_reference_panel_rgb(before["rgb"], sam3d_overlay)
            cv2.imwrite(
                str(save_dir / "left_replay_object_sam3d_prior_overlay.png"),
                cv2.cvtColor(sam3d_overlay_with_reference, cv2.COLOR_RGB2BGR),
            )

    ref_mask = np.asarray(before["mask"]) > 0
    ys, xs = np.nonzero(ref_mask)
    if len(xs):
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        object_crop = before["rgb"][y0:y1, x0:x1]
    else:
        object_crop = before["rgb"]
    cv2.imwrite(str(save_dir / "object.png"), cv2.cvtColor(object_crop, cv2.COLOR_RGB2BGR))

    registration_summary = dict(result.debug.get("registration_summary") or {})
    registration_summary.setdefault("downsample_debug", {"reference": before_downsample, "current": after_downsample})
    with open(save_dir / "registration.json", "w", encoding="utf-8") as handle:
        json.dump(registration_summary, handle, indent=2, default=str)

    init_prior_payload = dict(init_debug_payload)
    init_prior_payload["request_id"] = after.get("request", {}).get("request_id") or pair["replay_dir"].name
    init_prior_payload["applied"] = _reconstruct_tapir_transform(init_raw) is not None
    init_prior_payload["ransac"] = {
        "ok": bool(init_prior_payload.get("ransac_success", False)),
        "reason": init_prior_payload.get("reason", "ok" if init_prior_payload.get("success") else ""),
        "iters": int(init_prior_payload.get("ransac_num_hypotheses", 0) or 0),
        "sample_size": 4,
        "threshold_m": float(init_prior_payload.get("ransac_thresh_m", init_prior_payload.get("rmse_m", 0.0)) or 0.0),
    }
    with open(save_dir / f"{init_method}_prior_debug.json", "w", encoding="utf-8") as handle:
        json.dump(init_prior_payload, handle, indent=2, default=str)

    result_payload = {
        "case": {
            "case_name": _parse_replay_dir_name(pair["replay_dir"].name),
            "before_id": pair["before_dir"].name,
            "replay_id": pair["replay_dir"].name,
            "memory_dir": pair.get("memory_dir", ""),
        },
        "success": result.success,
        "quality": (result.debug.get("quality") or {}).get("quality", ""),
        "elapsed_s": result.elapsed_s,
        "init_method": init_method,
        "T_registration": np.asarray(result.T_registration, dtype=np.float64).tolist(),
        "T_slip": np.asarray(result.T_slip, dtype=np.float64).tolist(),
        "metrics": {
            "best_init": result.debug.get("best_init"),
            "best_score": result.debug.get("best_score"),
            "gicp_fitness": result.debug.get("reg_gicp_fitness"),
            "gicp_rmse_m": result.debug.get("reg_gicp_inlier_rmse"),
            "colored_icp_fitness": result.debug.get("reg_colored_icp_fitness"),
            "colored_icp_rmse_m": result.debug.get("reg_colored_icp_inlier_rmse"),
            "visibility_score": result.debug.get("reg_visibility_score"),
            "visible_count": result.debug.get("reg_visible_count"),
            "depth_ok_ratio": result.debug.get("reg_depth_ok_ratio"),
            "projection_iou": result.debug.get("reg_projection_iou"),
            "projection_precision": result.debug.get("reg_projection_precision"),
            "projection_recall": result.debug.get("reg_projection_recall"),
        },
        "tapir_init": result.debug.get("tapir_init"),
        "sam3d_init": result.debug.get("sam3d_init"),
        "registration": result.debug.get("registration_summary"),
        "stage_timings_s": result.debug.get("stage_timings_s"),
    }
    with open(save_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, default=str)

    response_payload = {
        "ok": bool(result.success),
        "request_id": after.get("request", {}).get("request_id") or pair["replay_dir"].name,
        "artifacts_dir": str(save_dir),
        "memory_dir": pair.get("memory_dir", ""),
        "experiment_config": after.get("experiment_config", {}) or {},
        "timings_ms": {
            "total": int(round(float(result.elapsed_s) * 1000.0)),
            "init": int(round(float(((result.debug.get("stage_timings_s") or {}).get(f"{init_stage_prefix} / init", 0.0)) * 1000.0))),
        },
        "selected_variant": (after.get("selected_replay_pose", {}) or {}).get("variant_name"),
        "overlay_path": str(save_dir / "left_replay_overlay.png"),
        "registration": result.debug.get("registration_summary"),
        "timings_detail_ms": {
            "service": {
                "init_method": init_method,
                "init": int(round(float(((result.debug.get("stage_timings_s") or {}).get(f"{init_stage_prefix} / init", 0.0)) * 1000.0))),
                "init_no_remote_roundtrip": int(
                    round(float(((result.debug.get("stage_timings_s") or {}).get(f"{init_stage_prefix} / init no_remote_roundtrip", 0.0)) * 1000.0))
                ),
                "init_remote_roundtrip": int(
                    round(float(((result.debug.get("stage_timings_s") or {}).get(f"{init_stage_prefix} / remote_roundtrip", 0.0)) * 1000.0))
                ),
                "init_infer": int(round(float(((result.debug.get("stage_timings_s") or {}).get(f"{init_stage_prefix} / infer", 0.0)) * 1000.0))),
                "init_prior_eval": int(round(float(((result.debug.get("stage_timings_s") or {}).get(f"{init_stage_prefix} / prior_eval", 0.0)) * 1000.0))),
                "init_completion_registration": int(round(float(((result.debug.get("stage_timings_s") or {}).get(f"{init_stage_prefix} / completion_registration", 0.0)) * 1000.0))),
                f"{init_method}_prior": int(round(float(((result.debug.get("stage_timings_s") or {}).get(f"{init_stage_prefix} / init", 0.0)) * 1000.0))),
                "registration": int(round(float(((result.debug.get("stage_timings_s") or {}).get("registration / total", 0.0)) * 1000.0))),
                "registration_prep": int(round(float(((result.debug.get("stage_timings_s") or {}).get("registration / prep", 0.0)) * 1000.0))),
                "registration_refine": int(round(float(((result.debug.get("stage_timings_s") or {}).get("registration / refine", 0.0)) * 1000.0))),
                "total": int(round(float(result.elapsed_s) * 1000.0)),
            }
        },
    }
    with open(save_dir / "response.json", "w", encoding="utf-8") as handle:
        json.dump(response_payload, handle, indent=2, default=str)


def _merge_tapir_cfg(base_cfg: TapirMatchConfig, replay_sample: dict[str, Any]) -> TapirMatchConfig:
    exp = replay_sample.get("experiment_config", {}) or {}
    if not isinstance(exp, dict) or not exp:
        return base_cfg
    dense_eval_max_points = base_cfg.dense_eval_max_points
    projection_eval_downsample = base_cfg.projection_eval_downsample
    dense_eval_raw = exp.get("tapir_dense_eval_max_points")
    proj_downsample_raw = exp.get("tapir_projection_eval_downsample")
    if dense_eval_raw is not None:
        try:
            dense_eval_max_points = int(dense_eval_raw)
        except Exception:
            dense_eval_max_points = base_cfg.dense_eval_max_points
    if proj_downsample_raw is not None:
        try:
            projection_eval_downsample = max(1, int(proj_downsample_raw))
        except Exception:
            projection_eval_downsample = base_cfg.projection_eval_downsample
    return replace(
        base_cfg,
        requested_points=int(exp.get("tapir_num_points", base_cfg.requested_points) or base_cfg.requested_points),
        max_query_points=int(exp.get("tapir_num_points", base_cfg.max_query_points) or base_cfg.max_query_points),
        random_seed=int(exp.get("tapir_random_seed", base_cfg.random_seed) or base_cfg.random_seed),
        prior_confidence_accept_thresh=float(
            exp.get("tapir_local_prior_confidence_accept_thresh", base_cfg.prior_confidence_accept_thresh)
        ),
        prior_confidence_strong_thresh=float(
            exp.get("tapir_local_prior_confidence_strong_thresh", base_cfg.prior_confidence_strong_thresh)
        ),
        dense_eval_max_points=int(dense_eval_max_points),
        projection_eval_downsample=int(projection_eval_downsample),
    )


def _merge_registration_cfg(base_cfg: RegistrationConfig, replay_sample: dict[str, Any]) -> RegistrationConfig:
    return base_cfg


def run_dof_cached_pair(
    before_dir: str | os.PathLike[str],
    replay_dir: str | os.PathLike[str],
    save_dir: str | os.PathLike[str],
    memory_dir: str = "",
    voxel_size: float = 0.003,
    tapir_cfg: TapirMatchConfig | None = None,
    sam3d_cfg: Sam3dMatchConfig | None = None,
    init_method: str = "tapir",
    log_fn=None,
) -> dict[str, Any]:
    if callable(log_fn):
        log = log_fn
    else:
        def log(message: str) -> None:
            print(message, flush=True)
    before_path = Path(before_dir).expanduser().resolve()
    replay_path = Path(replay_dir).expanduser().resolve()
    out_path = Path(save_dir).expanduser().resolve()
    pair = {
        "before_dir": before_path,
        "replay_dir": replay_path,
        "log_root": replay_path.parent.parent,
        "memory_dir": str(memory_dir or ""),
    }
    row = _result_row(pair, out_path, init_method=init_method)

    try:
        before = _load_dof_artifact_sample(before_path, is_replay=False)
        after = _load_dof_artifact_sample(replay_path, is_replay=True)
        cfg_used = _merge_registration_cfg(RegistrationConfig(voxel_size=float(voxel_size)), after)
        tapir_cfg_used = _merge_tapir_cfg(tapir_cfg or TapirMatchConfig(), after)
        tapir_cfg_used = replace(
            tapir_cfg_used,
            verbose_timings=True,
            log_fn=log,
            case_tag=_parse_replay_dir_name(replay_path.name),
        )
        result = run_reestimation(
            before=before,
            after=after,
            tapir_cfg=tapir_cfg_used,
            sam3d_cfg=sam3d_cfg,
            cfg=cfg_used,
            log_fn=log,
            init_method=init_method,
        )

        _write_pair_outputs(out_path, pair, before, after, result, cfg_used)
        _fill_result_row(row, result)
        return {"success": True, "row": row, "result": result, "save_dir": str(out_path)}
    except Exception as exc:
        out_path.mkdir(parents=True, exist_ok=True)
        row["error"] = str(exc)
        with open(out_path / f"{str(init_method or 'tapir')}_error.txt", "w", encoding="utf-8") as handle:
            handle.write(str(exc) + "\n\n" + traceback.format_exc())
        return {"success": False, "row": row, "error": str(exc), "save_dir": str(out_path)}


def _save_batch_summary(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "case_name",
        "init_method",
        "ok",
        "quality",
        "before_id",
        "replay_id",
        "elapsed_s",
        "best_init",
        "best_score",
        "init_success",
        "init_backend",
        "init_api_url",
        "init_matches",
        "init_valid3d",
        "init_inliers",
        "init_inlier_ratio",
        "init_rmse_m",
        "init_e2e_ms",
        "init_no_remote_roundtrip_ms",
        "init_no_rpc_io_ms",
        "tapir_success",
        "tapir_backend",
        "tapir_remote_host",
        "tapir_remote_port",
        "tapir_matches",
        "tapir_valid3d",
        "tapir_inliers",
        "tapir_inlier_ratio",
        "tapir_rmse_m",
        "tapir_track_e2e_ms",
        "tapir_track_no_remote_roundtrip_ms",
        "tapir_track_no_rpc_io_ms",
        "tapir_init_e2e_ms",
        "tapir_init_no_remote_roundtrip_ms",
        "tapir_init_no_rpc_io_ms",
        "visibility_score",
        "visible_count",
        "depth_ok_ratio",
        "gicp_fitness",
        "gicp_rmse_m",
        "colored_icp_fitness",
        "colored_icp_rmse_m",
        "save_dir",
        "error",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_dof_batch(
    dof_match_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
    start: int = 0,
    limit: int | None = None,
    voxel_size: float = 0.003,
    tapir_cfg: TapirMatchConfig | None = None,
    sam3d_cfg: Sam3dMatchConfig | None = None,
    init_method: str = "tapir",
    log_fn=None,
) -> dict[str, Any]:
    if callable(log_fn):
        log = log_fn
    else:
        def log(message: str) -> None:
            print(message, flush=True)
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    pairs = _iter_dof_pairs(dof_match_dir)
    if start > 0:
        pairs = pairs[start:]
    if limit is not None and limit > 0:
        pairs = pairs[:limit]

    rows: list[dict[str, Any]] = []
    used_output_names: set[str] = set()
    log(f"Found {len(pairs)} DOF replay pairs")
    for index, pair in enumerate(pairs, start=1):
        before_dir = pair.get("before_dir")
        replay_dir = pair["replay_dir"]
        pair_save_dir = out_root / _case_output_dirname(pair, used_output_names)
        if before_dir is None:
            row = _result_row(pair, pair_save_dir, init_method=init_method)
            row["error"] = f"Cannot resolve memory_dir: {pair.get('memory_dir', '')}"
            rows.append(row)
            log(f"[{index}/{len(pairs)}] SKIP {replay_dir.name}: {row['error']}")
            continue

        log(f"[{index}/{len(pairs)}] {before_dir.name} -> {replay_dir.name}")
        ret = run_dof_cached_pair(
            before_dir=before_dir,
            replay_dir=replay_dir,
            save_dir=pair_save_dir,
            memory_dir=pair.get("memory_dir", ""),
            voxel_size=voxel_size,
            tapir_cfg=tapir_cfg,
            sam3d_cfg=sam3d_cfg,
            init_method=init_method,
            log_fn=log,
        )
        row = ret["row"]
        rows.append(row)
        status = "OK" if ret.get("success") else "FAIL"
        detail = row.get("quality") or row.get("error") or ""
        log(f"  {status}: {detail} save={ret.get('save_dir')}")
        if ret.get("success") and ret.get("result") is not None:
            result = ret["result"]
            stage_timings = result.debug.get("stage_timings_s") or {}
            init_timings = (
                (result.debug.get("tapir_init") or {}).get("timings_s")
                if str(result.debug.get("init_method") or "") == "tapir"
                else (result.debug.get("sam3d_init") or {}).get("timings_s")
            ) or {}
            log(
                "  timings(ms): "
                + " ".join(
                    f"{name}={float(value) * 1000.0:.1f}"
                    for name, value in stage_timings.items()
                )
            )
            if init_timings:
                log(
                    "  init_detail(ms): "
                    + " ".join(
                        f"{name}={float(value) * 1000.0:.1f}"
                        for name, value in init_timings.items()
                    )
                )

    csv_path = out_root / "batch_summary.csv"
    json_path = out_root / "batch_summary.json"
    _save_batch_summary(csv_path, rows)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, default=str)

    ok_count = sum(1 for row in rows if row.get("ok"))
    summary = {
        "success": True,
        "total": len(rows),
        "ok": ok_count,
        "failed": len(rows) - ok_count,
        "output_dir": str(out_root),
        "csv_path": str(csv_path),
        "json_path": str(json_path),
        "rows": rows,
    }
    log(
        "Summary written: "
        f"{summary['csv_path']} | {summary['json_path']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="DOF batch pipeline with TAPIR or SAM3D init")
    parser.add_argument("dof_match_dir", nargs="?", default=DEFAULT_DOF_MATCH_DIR, help="dof_match root directory")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="batch output directory")
    parser.add_argument("--start", type=int, default=0, help="start offset for batch cases")
    parser.add_argument("--limit", type=int, default=0, help="maximum number of cases to run; 0 means no limit")
    parser.add_argument("--voxel-size", type=float, default=0.003, help="point cloud voxel size in meters")
    parser.add_argument("--init-method", choices=("tapir", "sam3d"), default="tapir", help="init method to run")
    parser.add_argument("--tapir-host", default=DEFAULT_TAPIR_HOST, help="remote TAPIR host")
    parser.add_argument("--tapir-port", type=int, default=DEFAULT_TAPIR_PORT, help="remote TAPIR port")
    parser.add_argument("--sam3d-url", default=DEFAULT_SAM3D_URL, help="SAM3D infer service base URL")
    parser.add_argument("--sam3d-timeout", type=float, default=60.0, help="SAM3D request timeout in seconds")
    parser.add_argument("--sam3d-seed", type=int, default=0, help="SAM3D infer seed")
    parser.add_argument("--sam3d-crop-to-mask", action="store_true", help="crop SAM3D infer inputs to mask bbox")
    parser.add_argument("--sam3d-no-pointmap", action="store_true", help="disable sending dense pointmap to SAM3D")
    args = parser.parse_args()

    tapir_cfg = TapirMatchConfig(
        backend="remote_tapnet",
        remote_host=str(args.tapir_host),
        remote_port=int(args.tapir_port),
    )
    sam3d_cfg = Sam3dMatchConfig(
        api_url=str(args.sam3d_url),
        timeout_s=float(args.sam3d_timeout),
        infer_seed=int(args.sam3d_seed),
        crop_to_mask=bool(args.sam3d_crop_to_mask),
        use_pointmap=bool(not args.sam3d_no_pointmap),
    )
    summary = run_dof_batch(
        dof_match_dir=args.dof_match_dir,
        output_dir=args.output_dir,
        start=max(int(args.start), 0),
        limit=(int(args.limit) if int(args.limit) > 0 else None),
        voxel_size=float(args.voxel_size),
        tapir_cfg=tapir_cfg,
        sam3d_cfg=sam3d_cfg,
        init_method=str(args.init_method),
    )
    print(
        f"{str(args.init_method).upper()} batch done: ok={summary['ok']} failed={summary['failed']} "
        f"total={summary['total']} output={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
