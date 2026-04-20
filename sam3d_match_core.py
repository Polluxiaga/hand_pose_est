from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pcd_registration_core import RegistrationConfig
from tapir_match_core import (
    _dense_transform_rmse,
    _projected_surface_metrics,
    _surface_points_from_mask_depth,
)

try:
    from sam3d.pointcloud_viewer import (
        DEFAULT_SAM3D_URL as _SAM3D_VIEWER_DEFAULT_URL,
        _build_dense_from_cloud_or_depth as _sam3d_build_dense_from_cloud_or_depth,
    )
    _SAM3D_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    _SAM3D_VIEWER_DEFAULT_URL = "http://101.132.143.105:5083"
    _SAM3D_IMPORT_ERROR = exc
    _sam3d_build_dense_from_cloud_or_depth = None

_SAM3D_REF_CODE_DIR = Path(__file__).resolve().parent / "sam3d" / "ref_code"
if _SAM3D_REF_CODE_DIR.is_dir() and str(_SAM3D_REF_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_SAM3D_REF_CODE_DIR))

try:
    from placedof_socket_service import (
        DEFAULT_SAM3_URL as _SAM3D_PICK_DEFAULT_URL,
        _finalize_pick_pose as _sam3d_finalize_pick_pose,
        _fit_local_plane_from_bbox as _sam3d_fit_local_plane_from_bbox,
        _fit_pick_object_box as _sam3d_fit_pick_object_box,
        _render_pick_overlay as _sam3d_render_pick_overlay,
        _run_qwen_sam3_target as _sam3d_run_qwen_sam3_target,
        _transform_dense_map_cam_to_base as _sam3d_transform_dense_map_cam_to_base,
    )
    _SAM3D_REF_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    _SAM3D_PICK_DEFAULT_URL = _SAM3D_VIEWER_DEFAULT_URL
    _SAM3D_REF_IMPORT_ERROR = exc
    _sam3d_finalize_pick_pose = None
    _sam3d_fit_local_plane_from_bbox = None
    _sam3d_fit_pick_object_box = None
    _sam3d_render_pick_overlay = None
    _sam3d_run_qwen_sam3_target = None
    _sam3d_transform_dense_map_cam_to_base = None

DEFAULT_SAM3D_URL = os.getenv("SAM3D_URL", _SAM3D_PICK_DEFAULT_URL)
DEFAULT_SAM3D_PICK_PROMPT = (
    "Find the single rigid object being held by the hand. Return one bbox around the ENTIRE visible object silhouette, "
    "not just the grasped top part or finger contact region. The object may extend far below the fingers. Include all "
    "visible connected parts that belong to the same object. For connected Lego assemblies, include the whole connected "
    "assembly, including lower protrusions and side bricks. For containers, bowls, bins, buckets, or cups, include the "
    "full visible body from rim to bottom. For arch or bridge shaped objects, include the full top bar and both visible "
    "legs. Prefer a slightly larger bbox that contains the whole object over a smaller bbox that crops any part of it. "
    "Exclude the hand, fingers, palm, wrist, sleeve, table, and background. Return JSON only."
)


@dataclass
class Sam3dMatchConfig:
    api_url: str = DEFAULT_SAM3D_URL
    timeout_s: float = 60.0
    infer_seed: int = 0
    use_pointmap: bool = True
    crop_to_mask: bool = False
    retry_count: int = 2
    retry_backoff_s: float = 1.0
    prior_confidence_accept_thresh: float = 0.45
    prior_confidence_strong_thresh: float = 0.72
    dense_eval_max_points: int = 2048
    projection_eval_downsample: int = 4
    completion_max_points: int = 8000


@dataclass
class Sam3dInitResult:
    success: bool
    T_before_cam_to_after_cam: np.ndarray | None = None
    debug: dict[str, Any] = field(default_factory=dict)
    before_pick_overlay_rgb: np.ndarray | None = None
    after_pick_overlay_rgb: np.ndarray | None = None


def _emit_timing_log(log_fn, message: str) -> None:
    if not callable(log_fn):
        return
    stamp = time.strftime("%H:%M:%S")
    log_fn(f"[{stamp}] {message}")


def _bbox_xyxy_from_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask_bool = np.asarray(mask) > 0
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]
    ys, xs = np.nonzero(mask_bool)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _rotation_angle_deg(R: np.ndarray) -> float:
    rot = np.asarray(R, dtype=np.float64).reshape(3, 3)
    val = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(val)))


def _transform_from_axes(
    ref_axes: np.ndarray,
    ref_center: np.ndarray,
    cur_axes: np.ndarray,
    cur_center: np.ndarray,
) -> np.ndarray:
    R = np.asarray(cur_axes, dtype=np.float64).reshape(3, 3) @ np.asarray(ref_axes, dtype=np.float64).reshape(3, 3).T
    t = np.asarray(cur_center, dtype=np.float64).reshape(3) - (R @ np.asarray(ref_center, dtype=np.float64).reshape(3))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _resolve_sample_extrinsics(
    sample: dict[str, Any],
    fallback_sample: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    extrinsics = sample.get("extrinsics") or {}
    if isinstance(extrinsics, dict):
        rotation = np.asarray(extrinsics.get("rotation"), dtype=np.float32)
        shift = np.asarray(extrinsics.get("shift"), dtype=np.float32)
        if rotation.shape == (3, 3) and shift.shape == (3,) and np.all(np.isfinite(rotation)) and np.all(np.isfinite(shift)):
            return rotation.astype(np.float32, copy=False), shift.astype(np.float32, copy=False), "sample"
    if fallback_sample is not None:
        fallback_extrinsics = fallback_sample.get("extrinsics") or {}
        if isinstance(fallback_extrinsics, dict):
            rotation = np.asarray(fallback_extrinsics.get("rotation"), dtype=np.float32)
            shift = np.asarray(fallback_extrinsics.get("shift"), dtype=np.float32)
            if rotation.shape == (3, 3) and shift.shape == (3,) and np.all(np.isfinite(rotation)) and np.all(np.isfinite(shift)):
                return rotation.astype(np.float32, copy=False), shift.astype(np.float32, copy=False), "fallback"
    raise RuntimeError("Missing valid camera extrinsics for SAM3D pick-pose adaptation.")


def _camera_pose_from_base_pose(
    center_base: np.ndarray,
    axes_base: np.ndarray,
    rotation_cam_to_base: np.ndarray,
    shift_cam_to_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    center_b = np.asarray(center_base, dtype=np.float64).reshape(3)
    axes_b = np.asarray(axes_base, dtype=np.float64).reshape(3, 3)
    rot_cb = np.asarray(rotation_cam_to_base, dtype=np.float64).reshape(3, 3)
    shift_cb = np.asarray(shift_cam_to_base, dtype=np.float64).reshape(3)
    center_cam = (center_b.reshape(1, 3) - shift_cb.reshape(1, 3)) @ rot_cb
    axes_cam = rot_cb.T @ axes_b
    if np.linalg.det(axes_cam) < 0.0:
        axes_cam[:, -1] *= -1.0
    return center_cam.reshape(3).astype(np.float32), axes_cam.astype(np.float32)


def _dense_map_cam_from_artifact(sample: dict[str, Any]) -> np.ndarray:
    dense_raw = np.asarray(sample.get("dense_raw"), dtype=np.float32)
    if dense_raw.size == 0:
        artifact_dir = Path(sample["artifact_dir"])
        dense_map_path = artifact_dir / "dense_map_cam.npy"
        if not dense_map_path.is_file():
            raise FileNotFoundError(f"Missing dense_map_cam.npy: {artifact_dir}")
        dense_raw = np.asarray(np.load(str(dense_map_path)), dtype=np.float32)
    if _sam3d_build_dense_from_cloud_or_depth is not None:
        return _sam3d_build_dense_from_cloud_or_depth(
            dense_raw,
            (int(sample["rgb"].shape[1]), int(sample["rgb"].shape[0])),
            sample["K"].K.astype(np.float32),
        )
    dense_map = np.asarray(dense_raw, dtype=np.float32)
    if dense_map.ndim != 3 or dense_map.shape[2] != 3:
        raise RuntimeError("SAM3D direct pick path requires a dense xyz map or pointcloud-viewer adapter.")
    return dense_map


def _serialize_sam3d_pick_pose(pose: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": str(pose.get("label", "")),
        "prompt": str(pose.get("prompt", "")),
        "bbox": [int(v) for v in pose.get("bbox", [])],
        "mask_area_px": int(pose.get("mask_area_px", 0)),
        "pixel": [int(v) for v in pose.get("pixel", [])],
        "pick_xquat": [float(v) for v in np.asarray(pose.get("pick_xquat", np.zeros(7)), dtype=np.float32).tolist()],
        "center_base": [float(v) for v in np.asarray(pose.get("center_base", np.zeros(3)), dtype=np.float32).tolist()],
        "axes_base": np.asarray(pose.get("axes_base", np.eye(3)), dtype=np.float32).tolist(),
        "size_xyz": [float(v) for v in np.asarray(pose.get("size_xyz", np.zeros(3)), dtype=np.float32).tolist()],
        "center_cam": [float(v) for v in np.asarray(pose.get("center_cam", np.zeros(3)), dtype=np.float32).tolist()],
        "axes_cam": np.asarray(pose.get("axes_cam", np.eye(3)), dtype=np.float32).tolist(),
        "grasp_origin_local": [float(v) for v in np.asarray(pose.get("grasp_origin_local", np.zeros(3)), dtype=np.float32).tolist()],
        "grasp_axes_local": np.asarray(pose.get("grasp_axes_local", np.eye(3)), dtype=np.float32).tolist(),
        "debug": dict(pose.get("debug") or {}),
        "timings_ms": dict(pose.get("timings_ms") or {}),
        "extrinsics_source": str(pose.get("extrinsics_source", "")),
        "grounding_source": str(pose.get("grounding_source", "")),
    }


def _run_sam3d_pick_pose(
    sample: dict[str, Any],
    cfg: Sam3dMatchConfig,
    rotation_cam_to_base: np.ndarray,
    shift_cam_to_base: np.ndarray,
    tag: str,
) -> dict[str, Any]:
    if (
        _sam3d_run_qwen_sam3_target is None
        or _sam3d_fit_local_plane_from_bbox is None
        or _sam3d_fit_pick_object_box is None
        or _sam3d_finalize_pick_pose is None
        or _sam3d_transform_dense_map_cam_to_base is None
    ):
        raise RuntimeError("SAM3D ref-code pick functions are unavailable.")
    t_total = time.perf_counter()
    t_stage = time.perf_counter()
    dense_map_cam = _dense_map_cam_from_artifact(sample)
    dense_map_base = _sam3d_transform_dense_map_cam_to_base(dense_map_cam, rotation_cam_to_base, shift_cam_to_base)
    dense_prepare_ms = int(round((time.perf_counter() - t_stage) * 1000.0))
    prompt = str(sample.get("object_prompt") or DEFAULT_SAM3D_PICK_PROMPT).strip() or DEFAULT_SAM3D_PICK_PROMPT
    target = {
        "order": 0,
        "target_id": str(tag),
        "role": "pick",
        "prompt": prompt,
        "offset": [0.0, 0.0, 0.0],
        "shift_prop": 0.5,
    }
    grounding_source = "qwen_sam3"
    t_stage = time.perf_counter()
    try:
        stage_result = _sam3d_run_qwen_sam3_target(
            target=target,
            left=Image.fromarray(np.asarray(sample["rgb"], dtype=np.uint8), mode="RGB"),
            rotation_cam_to_base=np.asarray(rotation_cam_to_base, dtype=np.float32),
            sam3_url=str(cfg.api_url),
            timeout_s=float(cfg.timeout_s),
        )
    except Exception as exc:
        mask_bool = np.asarray(sample.get("mask", np.zeros((0, 0), dtype=np.uint8))) > 0
        bbox = _bbox_xyxy_from_mask(mask_bool)
        if bbox is None:
            raise RuntimeError(f"SAM3D grounding failed and no cached mask bbox is available: {exc}") from exc
        center_pixel = [int(round((bbox[0] + bbox[2]) * 0.5)), int(round((bbox[1] + bbox[3]) * 0.5))]
        stage_result = {
            "order": 0,
            "target_id": str(tag),
            "role": "pick",
            "prompt": prompt,
            "label": "object",
            "orientation": None,
            "need_tracking": False,
            "visual_mask": None,
            "offset": [0.0, 0.0, 0.0],
            "shift_prop": 0.5,
            "bbox": [int(v) for v in bbox],
            "center_pixel": center_pixel,
            "mask": mask_bool,
            "dof_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "dof_r_square": 0.0,
            "qwen_status": f"fallback_cached_mask:{exc}",
            "timings_ms": {"qwen": 0, "sam3": 0, "total": 0},
        }
        grounding_source = "cached_mask_bbox"
    grounding_ms = int(round((time.perf_counter() - t_stage) * 1000.0))
    t_stage = time.perf_counter()
    plane_model = _sam3d_fit_local_plane_from_bbox(
        dense_map_base,
        tuple(int(v) for v in stage_result["bbox"]),
        np.asarray(stage_result["mask"], dtype=bool),
    )
    obj_fit = _sam3d_fit_pick_object_box(
        dense_map_base,
        tuple(int(v) for v in stage_result["bbox"]),
        np.asarray(stage_result["mask"], dtype=bool),
        plane_model,
    )
    pick_pose = _sam3d_finalize_pick_pose(
        stage_result,
        dense_map_cam,
        np.asarray(rotation_cam_to_base, dtype=np.float32),
        np.asarray(shift_cam_to_base, dtype=np.float32),
    )
    center_cam, axes_cam = _camera_pose_from_base_pose(
        obj_fit["center_base"],
        obj_fit["axes_base"],
        rotation_cam_to_base,
        shift_cam_to_base,
    )
    pose_fit_ms = int(round((time.perf_counter() - t_stage) * 1000.0))
    core_total_ms = int(round((time.perf_counter() - t_total) * 1000.0))
    overlay_rgb = None
    overlay_render_ms = 0
    if _sam3d_render_pick_overlay is not None:
        t_overlay = time.perf_counter()
        overlay_image = _sam3d_render_pick_overlay(
            Image.fromarray(np.asarray(sample["rgb"], dtype=np.uint8), mode="RGB"),
            {
                "mask": np.asarray(stage_result["mask"], dtype=bool),
                "bbox": [int(v) for v in stage_result["bbox"]],
                "label": str(stage_result["label"]),
                "pick_fit": {
                    "center_base": [float(v) for v in np.asarray(obj_fit["center_base"], dtype=np.float32).tolist()],
                    "axes_base": np.asarray(obj_fit["axes_base"], dtype=np.float32).tolist(),
                    "size_xyz": [float(v) for v in np.asarray(obj_fit["size_xyz"], dtype=np.float32).tolist()],
                    "grasp_origin_base": [float(v) for v in np.asarray(pick_pose["xquat"][:3], dtype=np.float32).tolist()],
                    "grasp_axes_base": np.asarray(obj_fit["axes_base"], dtype=np.float32).tolist(),
                },
            },
            sample["K"].K.astype(np.float32),
            np.asarray(rotation_cam_to_base, dtype=np.float32),
            np.asarray(shift_cam_to_base, dtype=np.float32),
        )
        overlay_rgb = np.asarray(overlay_image, dtype=np.uint8)
        overlay_render_ms = int(round((time.perf_counter() - t_overlay) * 1000.0))
    timings_ms = dict(stage_result.get("timings_ms") or {})
    timings_ms["dense_prepare"] = int(dense_prepare_ms)
    timings_ms["grounding_total"] = int(grounding_ms)
    timings_ms["pose_fit"] = int(pose_fit_ms)
    timings_ms["overlay_render"] = int(overlay_render_ms)
    timings_ms["total_core"] = int(core_total_ms)
    timings_ms["total"] = int(core_total_ms)
    return {
        "tag": str(tag),
        "prompt": prompt,
        "label": str(stage_result.get("label", "")),
        "bbox": [int(v) for v in stage_result["bbox"]],
        "mask": np.asarray(stage_result["mask"], dtype=bool),
        "mask_area_px": int(np.count_nonzero(stage_result["mask"])),
        "pixel": [int(v) for v in pick_pose["pixel"]],
        "pick_xquat": np.asarray(pick_pose["xquat"], dtype=np.float32),
        "center_base": np.asarray(obj_fit["center_base"], dtype=np.float32),
        "axes_base": np.asarray(obj_fit["axes_base"], dtype=np.float32),
        "size_xyz": np.asarray(obj_fit["size_xyz"], dtype=np.float32),
        "center_cam": center_cam.astype(np.float32, copy=False),
        "axes_cam": axes_cam.astype(np.float32, copy=False),
        "grasp_origin_local": np.asarray(obj_fit["grasp_origin_local"], dtype=np.float32),
        "grasp_axes_local": np.asarray(obj_fit["grasp_axes_local"], dtype=np.float32),
        "debug": dict(obj_fit.get("debug") or {}),
        "timings_ms": timings_ms,
        "grounding_source": grounding_source,
        "overlay_rgb": overlay_rgb,
    }


def estimate_sam3d_init_transform(
    before: dict[str, Any],
    after: dict[str, Any],
    cfg: Sam3dMatchConfig,
    log_fn=None,
) -> Sam3dInitResult:
    if _SAM3D_REF_IMPORT_ERROR is not None:
        raise RuntimeError(f"SAM3D ref-code import failed: {_SAM3D_REF_IMPORT_ERROR}")
    t_total = time.perf_counter()
    before_rot_cb, before_shift_cb, before_ext_source = _resolve_sample_extrinsics(before, fallback_sample=after)
    after_rot_cb, after_shift_cb, after_ext_source = _resolve_sample_extrinsics(after, fallback_sample=before)
    _emit_timing_log(log_fn, f"sam3d/pick before start extrinsics={before_ext_source}")
    before_pick = _run_sam3d_pick_pose(
        before,
        cfg=cfg,
        rotation_cam_to_base=before_rot_cb,
        shift_cam_to_base=before_shift_cb,
        tag="before_pick",
    )
    before_pick["extrinsics_source"] = before_ext_source
    _emit_timing_log(log_fn, f"sam3d/pick after start extrinsics={after_ext_source}")
    after_pick = _run_sam3d_pick_pose(
        after,
        cfg=cfg,
        rotation_cam_to_base=after_rot_cb,
        shift_cam_to_base=after_shift_cb,
        tag="after_pick",
    )
    after_pick["extrinsics_source"] = after_ext_source
    best_T = _transform_from_axes(
        before_pick["axes_cam"],
        before_pick["center_cam"],
        after_pick["axes_cam"],
        after_pick["center_cam"],
    )
    total_elapsed_s = time.perf_counter() - t_total
    debug: dict[str, Any] = {
        "enabled": True,
        "success": True,
        "method": "sam3d",
        "backend": "placedof_pick_obb",
        "api_url": str(cfg.api_url),
        "prior_rotation": best_T[:3, :3].tolist(),
        "prior_translation": best_T[:3, 3].tolist(),
        "before_pick": _serialize_sam3d_pick_pose(before_pick),
        "after_pick": _serialize_sam3d_pick_pose(after_pick),
        "rotation_change_deg": float(_rotation_angle_deg(best_T[:3, :3])),
        "translation_norm_m": float(np.linalg.norm(best_T[:3, 3])),
        "latency_ms": int(round(total_elapsed_s * 1000.0)),
        "timings_s": {
            "infer": float((before_pick["timings_ms"].get("total", 0) + after_pick["timings_ms"].get("total", 0)) / 1000.0),
            "before_pick": float(before_pick["timings_ms"].get("total", 0) / 1000.0),
            "after_pick": float(after_pick["timings_ms"].get("total", 0) / 1000.0),
            "remote_roundtrip": float(
                (before_pick["timings_ms"].get("grounding_total", 0) + after_pick["timings_ms"].get("grounding_total", 0))
                / 1000.0
            ),
            "completion_registration": 0.0,
            "total": float(total_elapsed_s),
        },
    }
    return Sam3dInitResult(
        success=True,
        T_before_cam_to_after_cam=best_T,
        debug=debug,
        before_pick_overlay_rgb=before_pick.get("overlay_rgb"),
        after_pick_overlay_rgb=after_pick.get("overlay_rgb"),
    )


def _sam3d_prior_confidence(
    projection_iou: float,
    projection_precision: float,
    fitness: float,
    relative_rmse: float,
) -> float:
    rmse_score = 0.0
    if np.isfinite(relative_rmse):
        rmse_score = float(np.clip(1.0 - (relative_rmse / 0.08), 0.0, 1.0))
    score = (
        0.36 * float(np.clip(projection_iou / 0.55, 0.0, 1.0))
        + 0.24 * float(np.clip(projection_precision / 0.75, 0.0, 1.0))
        + 0.24 * float(np.clip(fitness / 0.95, 0.0, 1.0))
        + 0.16 * rmse_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _sam3d_prior_confidence_components(
    projection_iou: float,
    projection_precision: float,
    fitness: float,
    relative_rmse: float,
) -> dict[str, float]:
    proj_score = float(np.clip(projection_iou / 0.55, 0.0, 1.0))
    precision_score = float(np.clip(projection_precision / 0.75, 0.0, 1.0))
    fitness_score = float(np.clip(fitness / 0.95, 0.0, 1.0))
    dense_score = 0.0
    if np.isfinite(relative_rmse):
        dense_score = float(np.clip(1.0 - (relative_rmse / 0.08), 0.0, 1.0))
    return {
        "prior_confidence": _sam3d_prior_confidence(
            projection_iou=projection_iou,
            projection_precision=projection_precision,
            fitness=fitness,
            relative_rmse=relative_rmse,
        ),
        "prior_confidence_proj_score": proj_score,
        "prior_confidence_dense_score": dense_score,
        "prior_confidence_ransac_score": fitness_score,
        "prior_confidence_inlier_score": precision_score,
        "prior_confidence_fitness_score": fitness_score,
        "prior_confidence_precision_score": precision_score,
    }


def evaluate_sam3d_prior(
    before: dict[str, Any],
    after: dict[str, Any],
    T_before_cam_to_after_cam: np.ndarray,
    cfg: Sam3dMatchConfig,
    reg_cfg: RegistrationConfig,
    log_fn=None,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    t_stage = time.perf_counter()
    dense_eval_max_points = max(256, int(getattr(cfg, "dense_eval_max_points", 2048) or 2048))
    ref_surface_points, ref_surface_debug = _surface_points_from_mask_depth(
        before["depth"],
        before["mask"],
        before["K"],
        reg_cfg.depth_max_m,
        dense_eval_max_points,
        int(getattr(cfg, "infer_seed", 0) or 0) + 151,
    )
    cur_surface_points, cur_surface_debug = _surface_points_from_mask_depth(
        after["depth"],
        after["mask"],
        after["K"],
        reg_cfg.depth_max_m,
        dense_eval_max_points,
        int(getattr(cfg, "infer_seed", 0) or 0) + 173,
    )
    timings["surface_points"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    prior_dense_rmse, prior_dense_fitness = _dense_transform_rmse(
        ref_surface_points,
        cur_surface_points,
        np.asarray(T_before_cam_to_after_cam, dtype=np.float64).reshape(4, 4),
        inlier_thresh_m=max(float(reg_cfg.gicp_max_corr_dist), 0.015),
    )
    timings["dense_eval_rmse"] = time.perf_counter() - t_stage
    ref_diag = float(np.linalg.norm(np.ptp(ref_surface_points, axis=0))) if len(ref_surface_points) else 0.0
    cur_diag = float(np.linalg.norm(np.ptp(cur_surface_points, axis=0))) if len(cur_surface_points) else 0.0
    dense_scale = max(ref_diag, cur_diag, 1e-6)
    rel_dense_rmse = float(prior_dense_rmse / dense_scale) if np.isfinite(prior_dense_rmse) else math.inf

    t_stage = time.perf_counter()
    proj_metrics = _projected_surface_metrics(
        ref_surface_points,
        np.asarray(T_before_cam_to_after_cam, dtype=np.float64).reshape(4, 4),
        after["mask"],
        after["K"],
        reg_cfg.depth_max_m,
        int(getattr(cfg, "projection_eval_downsample", 4) or 4),
    )
    timings["projection_eval"] = time.perf_counter() - t_stage

    confidence_debug = _sam3d_prior_confidence_components(
        projection_iou=float(proj_metrics.get("prior_proj_iou", 0.0)),
        projection_precision=float(proj_metrics.get("prior_proj_precision", 0.0)),
        fitness=float(prior_dense_fitness),
        relative_rmse=float(rel_dense_rmse),
    )
    prior_confidence = float(confidence_debug.get("prior_confidence", 0.0))
    prior_accept_thresh = float(getattr(cfg, "prior_confidence_accept_thresh", 0.45) or 0.45)
    prior_strong_thresh = float(getattr(cfg, "prior_confidence_strong_thresh", 0.72) or 0.72)
    borderline_override = bool(
        float(proj_metrics.get("prior_proj_iou", 0.0)) >= 0.44
        and float(proj_metrics.get("prior_proj_precision", 0.0)) >= 0.58
        and np.isfinite(rel_dense_rmse)
        and float(rel_dense_rmse) <= 0.050
        and float(prior_dense_fitness) >= 0.70
    )
    prior_accept = bool(prior_confidence >= prior_accept_thresh or borderline_override)
    prior_strong = bool(prior_confidence >= prior_strong_thresh)
    total_prior_eval_s = float(sum(float(value) for value in timings.values()))
    _emit_timing_log(
        log_fn,
        "sam3d/prior-eval "
        f"surface_ms={timings['surface_points'] * 1000.0:.1f} "
        f"rmse_ms={timings['dense_eval_rmse'] * 1000.0:.1f} "
        f"proj_ms={timings['projection_eval'] * 1000.0:.1f} "
        f"conf={prior_confidence:.3f} proj_iou={float(proj_metrics.get('prior_proj_iou', 0.0)):.3f} "
        f"prec={float(proj_metrics.get('prior_proj_precision', 0.0)):.3f} "
        f"rel_rmse={rel_dense_rmse:.4f}",
    )
    return {
        "prior_dense_rmse_eval": float(prior_dense_rmse),
        "prior_dense_relative_rmse_eval": float(rel_dense_rmse),
        "prior_dense_scale_m": float(dense_scale),
        "prior_dense_ref_diag_m": float(ref_diag),
        "prior_dense_cur_diag_m": float(cur_diag),
        "prior_dense_fitness_eval": float(prior_dense_fitness),
        "ref_surface_points": int(ref_surface_debug.get("used_points", 0)),
        "cur_surface_points": int(cur_surface_debug.get("used_points", 0)),
        "ref_surface_debug": ref_surface_debug,
        "cur_surface_debug": cur_surface_debug,
        **proj_metrics,
        **confidence_debug,
        "prior_confidence": float(prior_confidence),
        "prior_confidence_accept_thresh": float(prior_accept_thresh),
        "prior_confidence_strong_thresh": float(prior_strong_thresh),
        "prior_gate_proj_ok": bool(float(confidence_debug["prior_confidence_proj_score"]) >= 0.5),
        "prior_gate_rel_dense_ok": bool(float(confidence_debug["prior_confidence_dense_score"]) >= 0.5),
        "prior_gate_strong_3d_ok": bool(prior_strong),
        "prior_gate_decision": bool(prior_accept),
        "prior_gate_borderline_override": bool(borderline_override),
        "prior_gate_borderline_proj_iou_thresh": 0.44,
        "prior_gate_borderline_proj_precision_thresh": 0.58,
        "prior_gate_borderline_rel_dense_rmse_thresh": 0.050,
        "prior_gate_borderline_fitness_thresh": 0.70,
        "prior_gate_borderline_reason": ("borderline_projection_dense_fitness_override" if borderline_override else ""),
        "fallback_used": bool(not prior_accept),
        "num_inliers": 0,
        "inlier_count": 0,
        "timings_s": {
            "prior_eval": float(total_prior_eval_s),
            "surface_points": float(timings["surface_points"]),
            "dense_eval_rmse": float(timings["dense_eval_rmse"]),
            "projection_eval": float(timings["projection_eval"]),
        },
    }


__all__ = [
    "DEFAULT_SAM3D_PICK_PROMPT",
    "DEFAULT_SAM3D_URL",
    "Sam3dInitResult",
    "Sam3dMatchConfig",
    "estimate_sam3d_init_transform",
    "evaluate_sam3d_prior",
]
