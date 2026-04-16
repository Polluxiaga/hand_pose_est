from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Any

import cv2
import numpy as np

from match_geometry_core import (
    backproject as _backproject,
    mask_contains as _mask_contains,
    mask_distance_values as _mask_distance_values,
    object_projection_support as _object_projection_support,
    ransac_rigid_transform as _ransac_rigid_transform,
    roi_contains as _roi_contains,
    sample_depth as _sample_depth,
    select_valid_matches as _select_valid_matches,
)

@dataclass
class LightGlueMatchConfig:
    max_keypoints: int = 1024
    min_matches: int = 8
    min_inliers: int = 5
    min_inlier_ratio: float = 0.15
    ransac_iters: int = 1200
    ransac_thresh_m: float = 0.012
    max_ransac_rmse_m: float = 0.015
    max_depth_m: float = 2.0
    mask_margin_px: int = 8
    roi_bbox_margin_px: int = 36
    roi_near_mask_max_dist_px: float = 12.0
    min_near_mask_inlier_ratio: float = 0.25
    min_object_projection_near_mask_ratio: float = 0.10
    object_projection_sample_count: int = 1500
    use_roi_fallback: bool = True
    use_roi_crop: bool = True
    crop_bbox_margin_px: int = 56
    max_crop_long_edge: int = 0
    max_valid_matches: int = 240
    small_sample_enable: bool = True
    small_sample_max_matches: int = 24
    small_sample_min_inliers: int = 4
    small_sample_min_inlier_ratio: float = 0.25
    small_sample_ransac_thresh_m: float = 0.020
    small_sample_max_ransac_rmse_m: float = 0.018
    random_seed: int = 42
    device: str | None = None


@dataclass
class LightGlueInitResult:
    success: bool
    T_before_cam_to_after_cam: np.ndarray | None = None
    before_keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    after_keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    matches: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    valid_match_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    inlier_match_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    debug: dict[str, Any] = field(default_factory=dict)


_SUPERPOINT = None
_LIGHTGLUE = None
_MODEL_DEVICE = None
_TORCH = None
_LIGHTGLUE_CLASS = None
_SUPERPOINT_CLASS = None
_RBD = None
_IMPORT_ERROR: Exception | None = None


def _ensure_lightglue_imports():
    global _TORCH, _LIGHTGLUE_CLASS, _SUPERPOINT_CLASS, _RBD, _IMPORT_ERROR
    if _TORCH is not None and _LIGHTGLUE_CLASS is not None and _SUPERPOINT_CLASS is not None and _RBD is not None:
        return _TORCH, _LIGHTGLUE_CLASS, _SUPERPOINT_CLASS, _RBD
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
    except Exception as exc:
        _IMPORT_ERROR = exc
        raise ImportError(
            "SuperPoint-LightGlue dependencies are missing. Install with: "
            "pip install torch lightglue"
        ) from exc
    _TORCH = torch
    _LIGHTGLUE_CLASS = LightGlue
    _SUPERPOINT_CLASS = SuperPoint
    _RBD = rbd
    _IMPORT_ERROR = None
    return _TORCH, _LIGHTGLUE_CLASS, _SUPERPOINT_CLASS, _RBD


def _image_to_tensor_rgb(image_rgb: np.ndarray, device: str):
    torch, _, _, _ = _ensure_lightglue_imports()
    arr = np.asarray(image_rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return torch.from_numpy(gray)[None, None].to(device)


def _load_models(cfg: LightGlueMatchConfig):
    global _SUPERPOINT, _LIGHTGLUE, _MODEL_DEVICE
    torch, LightGlue, SuperPoint, _ = _ensure_lightglue_imports()
    if cfg.device:
        device = cfg.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if _SUPERPOINT is None or _LIGHTGLUE is None or _MODEL_DEVICE != device:
        _SUPERPOINT = SuperPoint(max_num_keypoints=int(cfg.max_keypoints)).eval().to(device)
        _LIGHTGLUE = LightGlue(features="superpoint").eval().to(device)
        _MODEL_DEVICE = device
    return _SUPERPOINT, _LIGHTGLUE, device


def _bbox_from_mask(mask: np.ndarray, margin_px: int) -> tuple[int, int, int, int] | None:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = m > 0
    if not m.any():
        return None
    h, w = m.shape[:2]
    ys, xs = np.nonzero(m)
    margin = max(int(margin_px), 0)
    x1 = max(int(xs.min()) - margin, 0)
    y1 = max(int(ys.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin + 1, w)
    y2 = min(int(ys.max()) + margin + 1, h)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_with_bbox(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> tuple[np.ndarray, np.ndarray]:
    if bbox is None:
        return image, np.array([0.0, 0.0], dtype=np.float32)
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2], np.array([float(x1), float(y1)], dtype=np.float32)


def _resize_for_matching(
    image: np.ndarray,
    max_long_edge: int,
) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    max_edge = max(h, w)
    if max_long_edge <= 0 or max_edge <= int(max_long_edge):
        return image, 1.0
    scale = float(max_long_edge) / float(max_edge)
    new_w = max(int(round(w * scale)), 1)
    new_h = max(int(round(h * scale)), 1)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def estimate_lightglue_init_transform(
    before_rgb: np.ndarray,
    before_depth_m: np.ndarray,
    before_mask: np.ndarray,
    after_rgb: np.ndarray,
    after_depth_m: np.ndarray,
    after_mask: np.ndarray,
    intr: Any,
    cfg: LightGlueMatchConfig | None = None,
) -> LightGlueInitResult:
    cfg = cfg or LightGlueMatchConfig()
    debug: dict[str, Any] = {"enabled": True}
    timings: dict[str, float] = {}
    t_total = time.perf_counter()
    try:
        t_stage = time.perf_counter()
        extractor, matcher, device = _load_models(cfg)
        torch, _, _, rbd = _ensure_lightglue_imports()
        timings["load_models"] = time.perf_counter() - t_stage
        debug["device"] = device
    except Exception as exc:
        timings["total"] = time.perf_counter() - t_total
        debug["timings_s"] = timings
        debug.update({"success": False, "error": str(exc), "stage": "load_models"})
        return LightGlueInitResult(success=False, debug=debug)

    try:
        with torch.inference_mode():  # type: ignore[union-attr]
            t_stage = time.perf_counter()
            before_match_rgb = before_rgb
            after_match_rgb = after_rgb
            offset0 = np.array([0.0, 0.0], dtype=np.float32)
            offset1 = np.array([0.0, 0.0], dtype=np.float32)
            crop0 = None
            crop1 = None
            if cfg.use_roi_crop:
                crop0 = _bbox_from_mask(before_mask, cfg.crop_bbox_margin_px)
                crop1 = _bbox_from_mask(after_mask, cfg.crop_bbox_margin_px)
                if crop0 is not None and crop1 is not None:
                    before_match_rgb, offset0 = _crop_with_bbox(before_rgb, crop0)
                    after_match_rgb, offset1 = _crop_with_bbox(after_rgb, crop1)
            before_match_rgb, scale0 = _resize_for_matching(before_match_rgb, cfg.max_crop_long_edge)
            after_match_rgb, scale1 = _resize_for_matching(after_match_rgb, cfg.max_crop_long_edge)
            debug["match_scope"] = "roi_crop" if crop0 is not None and crop1 is not None and cfg.use_roi_crop else "full_image"
            debug["before_crop_bbox"] = crop0
            debug["after_crop_bbox"] = crop1
            debug["before_match_shape"] = tuple(int(v) for v in before_match_rgb.shape[:2])
            debug["after_match_shape"] = tuple(int(v) for v in after_match_rgb.shape[:2])
            debug["before_match_scale"] = float(scale0)
            debug["after_match_scale"] = float(scale1)
            debug["max_keypoints"] = int(cfg.max_keypoints)
            timings["crop"] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            image0 = _image_to_tensor_rgb(before_match_rgb, device)
            image1 = _image_to_tensor_rgb(after_match_rgb, device)
            timings["to_tensor"] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            feats0 = extractor.extract(image0)
            feats1 = extractor.extract(image1)
            timings["superpoint_extract"] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            matches01 = matcher({"image0": feats0, "image1": feats1})
            timings["lightglue_match"] = time.perf_counter() - t_stage

            t_stage = time.perf_counter()
            feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]  # type: ignore[misc]
            kpts0 = feats0["keypoints"].detach().cpu().numpy().astype(np.float32)
            kpts1 = feats1["keypoints"].detach().cpu().numpy().astype(np.float32)
            kpts0 = kpts0 / max(float(scale0), 1e-8) + offset0.reshape(1, 2)
            kpts1 = kpts1 / max(float(scale1), 1e-8) + offset1.reshape(1, 2)
            matches = matches01["matches"].detach().cpu().numpy().astype(np.int32)
            timings["decode_matches"] = time.perf_counter() - t_stage
    except Exception as exc:
        timings["total"] = time.perf_counter() - t_total
        debug["timings_s"] = timings
        debug.update({"success": False, "error": str(exc), "stage": "match"})
        return LightGlueInitResult(success=False, debug=debug)

    t_stage = time.perf_counter()
    debug["num_keypoints_before"] = int(len(kpts0))
    debug["num_keypoints_after"] = int(len(kpts1))
    debug["num_matches"] = int(len(matches))
    if len(matches) < int(cfg.min_matches):
        timings["filter_matches"] = time.perf_counter() - t_stage
        timings["total"] = time.perf_counter() - t_total
        debug["timings_s"] = timings
        debug.update({"success": False, "reason": "too few raw matches"})
        return LightGlueInitResult(False, before_keypoints=kpts0, after_keypoints=kpts1, matches=matches, debug=debug)

    xy0 = kpts0[matches[:, 0]]
    xy1 = kpts1[matches[:, 1]]
    keep_mask_before = _mask_contains(before_mask, xy0, cfg.mask_margin_px)
    keep_mask_after = _mask_contains(after_mask, xy1, cfg.mask_margin_px)
    keep_mask = keep_mask_before & keep_mask_after
    keep_roi_before = _roi_contains(before_mask, xy0, cfg.roi_bbox_margin_px)
    keep_roi_after = _roi_contains(after_mask, xy1, cfg.roi_bbox_margin_px)
    keep_roi = keep_roi_before & keep_roi_after
    dist0 = _mask_distance_values(before_mask, xy0)
    dist1 = _mask_distance_values(after_mask, xy1)
    keep_near_mask_before = dist0 <= float(cfg.roi_near_mask_max_dist_px)
    keep_near_mask_after = dist1 <= float(cfg.roi_near_mask_max_dist_px)
    keep_near_mask = keep_near_mask_before & keep_near_mask_after
    keep_roi_near = keep_roi & keep_near_mask
    z0, valid0 = _sample_depth(before_depth_m, xy0, cfg.max_depth_m)
    z1, valid1 = _sample_depth(after_depth_m, xy1, cfg.max_depth_m)
    if cfg.use_roi_fallback:
        keep_region = keep_mask | keep_roi_near
        region_mode = "mask_or_near_mask"
    else:
        keep_region = keep_mask
        region_mode = "mask_only"
    keep = keep_region & valid0 & valid1
    match_weights = np.full(len(matches), 0.01, dtype=np.float64)
    match_weights[keep_roi] = 0.06
    match_weights[keep_roi_near] = 0.55
    match_weights[keep_mask] = 1.0
    valid_match_indices = _select_valid_matches(
        keep,
        match_weights,
        cfg.max_valid_matches,
        np.random.default_rng(int(cfg.random_seed) + 17),
    )
    timings["filter_matches"] = time.perf_counter() - t_stage
    debug["mask_margin_px"] = int(cfg.mask_margin_px)
    debug["roi_bbox_margin_px"] = int(cfg.roi_bbox_margin_px)
    debug["roi_near_mask_max_dist_px"] = float(cfg.roi_near_mask_max_dist_px)
    debug["region_mode"] = region_mode
    debug["num_before_mask_matches"] = int(keep_mask_before.sum())
    debug["num_after_mask_matches"] = int(keep_mask_after.sum())
    debug["num_mask_matches"] = int(keep_mask.sum())
    debug["num_before_roi_matches"] = int(keep_roi_before.sum())
    debug["num_after_roi_matches"] = int(keep_roi_after.sum())
    debug["num_roi_matches"] = int(keep_roi.sum())
    debug["num_before_near_mask_matches"] = int(keep_near_mask_before.sum())
    debug["num_after_near_mask_matches"] = int(keep_near_mask_after.sum())
    debug["num_near_mask_matches"] = int(keep_near_mask.sum())
    debug["num_roi_near_mask_matches"] = int(keep_roi_near.sum())
    debug["num_before_depth_matches"] = int(valid0.sum())
    debug["num_after_depth_matches"] = int(valid1.sum())
    debug["num_region_depth_matches"] = int(keep.sum())
    debug["num_valid_3d_matches"] = int(len(valid_match_indices))
    debug["num_valid_mask_3d_matches"] = int((keep_mask[valid_match_indices]).sum()) if len(valid_match_indices) else 0
    debug["num_valid_near_mask_3d_matches"] = int((keep_near_mask[valid_match_indices]).sum()) if len(valid_match_indices) else 0
    debug["num_valid_roi_only_3d_matches"] = int(((~keep_mask) & keep_roi)[valid_match_indices].sum()) if len(valid_match_indices) else 0
    debug["num_valid_far_roi_3d_matches"] = int(((~keep_near_mask) & keep_roi)[valid_match_indices].sum()) if len(valid_match_indices) else 0
    if len(valid_match_indices) < int(cfg.min_matches):
        timings["total"] = time.perf_counter() - t_total
        debug["timings_s"] = timings
        debug.update({"success": False, "reason": "too few mask/depth-valid matches"})
        return LightGlueInitResult(
            False, before_keypoints=kpts0, after_keypoints=kpts1, matches=matches,
            valid_match_indices=valid_match_indices, debug=debug,
        )

    t_stage = time.perf_counter()
    src3 = _backproject(xy0[valid_match_indices], z0[valid_match_indices], intr)
    tgt3 = _backproject(xy1[valid_match_indices], z1[valid_match_indices], intr)
    valid_weights = match_weights[valid_match_indices]
    timings["backproject"] = time.perf_counter() - t_stage

    t_stage = time.perf_counter()
    T, inliers_local, ransac_debug = _ransac_rigid_transform(src3, tgt3, cfg, weights=valid_weights)
    fallback_used = False
    if (
        cfg.small_sample_enable
        and len(valid_match_indices) <= int(cfg.small_sample_max_matches)
        and (T is None or int(inliers_local.sum()) < int(cfg.min_inliers))
    ):
        relaxed_cfg = replace(
            cfg,
            min_inliers=int(cfg.small_sample_min_inliers),
            ransac_thresh_m=float(cfg.small_sample_ransac_thresh_m),
        )
        T_relaxed, inliers_relaxed, relaxed_debug = _ransac_rigid_transform(
            src3, tgt3, relaxed_cfg, weights=valid_weights,
        )
        if T_relaxed is not None and int(inliers_relaxed.sum()) > int(inliers_local.sum()):
            T = T_relaxed
            inliers_local = inliers_relaxed
            ransac_debug = relaxed_debug
            fallback_used = True
    timings["ransac_3d"] = time.perf_counter() - t_stage
    inlier_match_indices = valid_match_indices[inliers_local].astype(np.int32)
    debug.update(ransac_debug)
    debug["small_sample_fallback_used"] = bool(fallback_used)
    debug["num_inliers"] = int(len(inlier_match_indices))
    debug["inlier_ratio"] = float(len(inlier_match_indices) / max(len(valid_match_indices), 1))
    debug["num_mask_inliers"] = int((keep_mask[inlier_match_indices]).sum()) if len(inlier_match_indices) else 0
    debug["num_near_mask_inliers"] = int((keep_near_mask[inlier_match_indices]).sum()) if len(inlier_match_indices) else 0
    debug["near_mask_inlier_ratio"] = float(debug["num_near_mask_inliers"] / max(len(inlier_match_indices), 1))
    debug["num_roi_only_inliers"] = int(((~keep_mask) & keep_roi)[inlier_match_indices].sum()) if len(inlier_match_indices) else 0
    debug["num_far_roi_inliers"] = int(((~keep_near_mask) & keep_roi)[inlier_match_indices].sum()) if len(inlier_match_indices) else 0
    rmse = float(debug.get("ransac_rmse_m", math.inf))
    if debug["num_mask_inliers"] > 0:
        debug["acceptance_mode"] = "mask_supported"
    elif debug["num_near_mask_inliers"] > 0:
        debug["acceptance_mode"] = "near_mask_supported"
    elif debug["num_roi_only_inliers"] > 0:
        debug["acceptance_mode"] = "roi_only_low_rmse"
    else:
        debug["acceptance_mode"] = "none"
    req_min_inliers = int(cfg.small_sample_min_inliers) if fallback_used else int(cfg.min_inliers)
    req_min_ratio = float(cfg.small_sample_min_inlier_ratio) if fallback_used else float(cfg.min_inlier_ratio)
    req_max_rmse = float(cfg.small_sample_max_ransac_rmse_m) if fallback_used else float(cfg.max_ransac_rmse_m)
    debug["required_min_inliers"] = req_min_inliers
    debug["required_min_inlier_ratio"] = req_min_ratio
    debug["required_max_ransac_rmse_m"] = req_max_rmse
    near_mask_ok = debug["near_mask_inlier_ratio"] >= float(cfg.min_near_mask_inlier_ratio)
    debug["required_min_near_mask_inlier_ratio"] = float(cfg.min_near_mask_inlier_ratio)
    debug["near_mask_gate_ok"] = bool(near_mask_ok)
    projection_ok = True
    if float(cfg.min_object_projection_near_mask_ratio) > 0.0:
        projection_ok = False
        if T is not None:
            projection_debug = _object_projection_support(
                before_depth_m=before_depth_m,
                before_mask=before_mask,
                after_mask=after_mask,
                T_before_to_after=T,
                intr=intr,
                max_depth_m=cfg.max_depth_m,
                near_mask_px=cfg.roi_near_mask_max_dist_px,
                max_samples=cfg.object_projection_sample_count,
                random_seed=int(cfg.random_seed) + 29,
            )
            debug.update(projection_debug)
            projection_ok = (
                float(projection_debug.get("object_projection_near_mask_ratio", 0.0))
                >= float(cfg.min_object_projection_near_mask_ratio)
            )
    debug["required_min_object_projection_near_mask_ratio"] = float(cfg.min_object_projection_near_mask_ratio)
    debug["object_projection_gate_ok"] = bool(projection_ok)
    debug["success"] = bool(
        T is not None
        and len(inlier_match_indices) >= req_min_inliers
        and debug["inlier_ratio"] >= req_min_ratio
        and rmse <= req_max_rmse
        and near_mask_ok
        and projection_ok
    )
    if not debug["success"]:
        if len(inlier_match_indices) < req_min_inliers:
            debug.setdefault("reason", "too few ransac inliers")
        elif debug["inlier_ratio"] < req_min_ratio:
            debug.setdefault("reason", "low ransac inlier ratio")
        elif rmse > req_max_rmse:
            debug.setdefault("reason", "high ransac rmse")
        elif not near_mask_ok:
            debug.setdefault("reason", "low near-mask inlier ratio")
        elif not projection_ok:
            debug.setdefault("reason", "low object projection near-mask ratio")
        else:
            debug.setdefault("reason", "feature transform rejected")
        T = None
    timings["total"] = time.perf_counter() - t_total
    debug["timings_s"] = timings
    return LightGlueInitResult(
        success=bool(debug["success"]),
        T_before_cam_to_after_cam=T,
        before_keypoints=kpts0,
        after_keypoints=kpts1,
        matches=matches,
        valid_match_indices=valid_match_indices,
        inlier_match_indices=inlier_match_indices,
        debug=debug,
    )


def draw_lightglue_matches(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    result: LightGlueInitResult,
    max_lines: int = 160,
) -> np.ndarray:
    left = np.asarray(before_rgb).copy()
    right = np.asarray(after_rgb).copy()
    if left.dtype != np.uint8:
        left = np.clip(left, 0, 255).astype(np.uint8)
    if right.dtype != np.uint8:
        right = np.clip(right, 0, 255).astype(np.uint8)
    h = max(left.shape[0], right.shape[0])
    canvas = np.zeros((h, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1]:] = right

    if len(result.matches) == 0:
        cv2.putText(canvas, "No LightGlue matches", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)
        return canvas

    def _match_color(match_idx: int, muted: bool = False) -> tuple[int, int, int]:
        hue = (float((int(match_idx) * 37) % 180) / 180.0)
        hsv = np.array([[[hue * 179.0, 210.0, 255.0]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)
        if muted:
            rgb = 0.45 * rgb + 0.55 * np.array([120.0, 120.0, 120.0], dtype=np.float32)
        return tuple(int(v) for v in np.clip(rgb, 0, 255))

    inlier_set = set(int(i) for i in result.inlier_match_indices.tolist())
    valid = result.valid_match_indices
    draw_indices = valid
    raw_mode = False
    if len(draw_indices) == 0:
        raw_mode = True
        draw_indices = np.arange(len(result.matches), dtype=np.int32)
    if len(draw_indices) > max_lines:
        rng = np.random.default_rng(7)
        draw_indices = rng.choice(draw_indices, max_lines, replace=False)
    for mi in draw_indices:
        a, b = result.matches[int(mi)]
        p0 = tuple(np.round(result.before_keypoints[a]).astype(int))
        p1_raw = np.round(result.after_keypoints[b]).astype(int)
        p1 = (int(p1_raw[0] + left.shape[1]), int(p1_raw[1]))
        is_in = int(mi) in inlier_set
        color = _match_color(int(mi), muted=(not raw_mode and not is_in))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, "Before", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "After", (left.shape[1] + 16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        (
            f"raw={len(result.matches)} mask/depth-valid={len(result.valid_match_indices)} "
            f"inliers={len(result.inlier_match_indices)} "
            f"{'multi-color raw matches' if raw_mode else 'bright=inlier muted=outlier'}"
        ),
        (16, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
