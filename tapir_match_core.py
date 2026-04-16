from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
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
class TapirMatchConfig:
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
    max_valid_matches: int = 240
    small_sample_enable: bool = True
    small_sample_max_matches: int = 24
    small_sample_min_inliers: int = 4
    small_sample_min_inlier_ratio: float = 0.25
    small_sample_ransac_thresh_m: float = 0.020
    small_sample_max_ransac_rmse_m: float = 0.018
    random_seed: int = 42
    backend: str = "auto"
    tapir_checkpoint_path: str = ""
    tapir_device: str | None = None
    tapir_resize_height: int = 256
    tapir_resize_width: int = 256
    tapir_query_chunk_size: int = 64
    max_query_points: int = 512
    min_track_quality: float = 0.01
    min_track_distance_px: int = 6
    lk_win_size: int = 21
    lk_max_level: int = 3
    max_lk_error: float = 32.0


@dataclass
class TapirInitResult:
    success: bool
    T_before_cam_to_after_cam: np.ndarray | None = None
    before_keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    after_keypoints: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))
    matches: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    valid_match_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    inlier_match_indices: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    debug: dict[str, Any] = field(default_factory=dict)


_ROOT = Path(__file__).resolve().parent
_TAPNET_ROOT = _ROOT / "tapnet"
_TAPIR_MODEL = None
_TAPIR_MODEL_KEY: tuple[str, str, bool] | None = None
_TORCH = None
_TORCH_F = None
_TAPIR_MODEL_MODULE = None
_TAPIR_IMPORT_ERROR: Exception | None = None


def _ensure_tapir_torch_imports():
    global _TORCH, _TORCH_F, _TAPIR_MODEL_MODULE, _TAPIR_IMPORT_ERROR
    if _TORCH is not None and _TORCH_F is not None and _TAPIR_MODEL_MODULE is not None:
        return _TORCH, _TORCH_F, _TAPIR_MODEL_MODULE
    if str(_TAPNET_ROOT) not in sys.path:
        sys.path.insert(0, str(_TAPNET_ROOT))
    try:
        import torch
        import torch.nn.functional as torch_f
        from tapnet.torch import tapir_model
    except Exception as exc:
        _TAPIR_IMPORT_ERROR = exc
        raise ImportError(f"TAPIR torch backend import failed: {exc}") from exc
    _TORCH = torch
    _TORCH_F = torch_f
    _TAPIR_MODEL_MODULE = tapir_model
    _TAPIR_IMPORT_ERROR = None
    return _TORCH, _TORCH_F, _TAPIR_MODEL_MODULE


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {arr.shape}")
    return arr


def _select_tapir_device(device: str | None):
    torch, _, _ = _ensure_tapir_torch_imports()
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _find_tapir_checkpoint(path: str) -> str:
    if path:
        ckpt = Path(path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"TAPIR checkpoint not found: {ckpt}")
        return str(ckpt)
    candidates = [
        _ROOT / "tapnet" / "checkpoints" / "tapir_checkpoint_panning.pt",
        _ROOT / "tapnet" / "checkpoints" / "tapir_checkpoint_v2.pt",
        _ROOT / "tapnet" / "checkpoints" / "bootstapir_checkpoint_v2.pt",
        _ROOT / "tapnet" / "tapnet" / "checkpoints" / "tapir_checkpoint_panning.pt",
        _ROOT / "tapnet" / "tapnet" / "checkpoints" / "tapir_checkpoint_v2.pt",
        _ROOT / "tapnet" / "tapnet" / "checkpoints" / "bootstapir_checkpoint_v2.pt",
        _ROOT / "checkpoints" / "tapir_checkpoint_panning.pt",
        _ROOT / "checkpoints" / "tapir_checkpoint_v2.pt",
        _ROOT / "checkpoints" / "bootstapir_checkpoint_v2.pt",
    ]
    for ckpt in candidates:
        if ckpt.is_file():
            return str(ckpt)
    raise FileNotFoundError(
        "No TAPIR PyTorch checkpoint found. Put tapir_checkpoint_panning.pt "
        "under tapnet/checkpoints/ or pass --tapir-checkpoint."
    )


def _load_tapir_model(checkpoint_path: str, device: str | None, causal: bool = False):
    global _TAPIR_MODEL, _TAPIR_MODEL_KEY
    torch, _, tapir_model = _ensure_tapir_torch_imports()
    device_obj = _select_tapir_device(device)
    resolved = _find_tapir_checkpoint(checkpoint_path)
    key = (resolved, str(device_obj), bool(causal))
    if _TAPIR_MODEL is not None and _TAPIR_MODEL_KEY == key:
        return _TAPIR_MODEL, device_obj, resolved
    try:
        state = torch.load(resolved, map_location=device_obj, weights_only=True)
    except TypeError:
        state = torch.load(resolved, map_location=device_obj)
    mixer_weight = state.get("torch_pips_mixer.linear.weight") if isinstance(state, dict) else None
    mixer_in_dim = int(mixer_weight.shape[1]) if mixer_weight is not None and hasattr(mixer_weight, "shape") else 535
    pyramid_level = max(int(round((mixer_in_dim - (4 + 128 + 256)) / 49.0)) - 2, 0)
    has_extra_convs = any(str(k).startswith("extra_convs.") for k in state.keys()) if isinstance(state, dict) else True
    model = tapir_model.TAPIR(
        pyramid_level=pyramid_level,
        extra_convs=has_extra_convs,
        use_casual_conv=bool(causal),
    )
    model.load_state_dict(state)
    model = model.to(device_obj).eval()
    _TAPIR_MODEL = model
    _TAPIR_MODEL_KEY = key
    return model, device_obj, resolved


def _preprocess_tapir_frames(frames):
    frames = frames.float()
    return frames / 255.0 * 2.0 - 1.0


def _tapir_track_confidence(occlusions, expected_dist):
    _, torch_f, _ = _ensure_tapir_torch_imports()
    return (1.0 - torch_f.sigmoid(occlusions)) * (1.0 - torch_f.sigmoid(expected_dist))


def _prepare_gray(image_rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _sample_query_points(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    cfg: TapirMatchConfig,
) -> np.ndarray:
    gray = _prepare_gray(image_rgb)
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m > 0).astype(np.uint8)
    valid_depth = np.isfinite(depth_m) & (depth_m > 1e-5) & (depth_m < float(cfg.max_depth_m))
    query_mask = (m > 0) & valid_depth
    if not query_mask.any():
        return np.zeros((0, 2), dtype=np.float32)

    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=int(cfg.max_query_points),
        qualityLevel=float(cfg.min_track_quality),
        minDistance=float(cfg.min_track_distance_px),
        mask=(query_mask.astype(np.uint8) * 255),
        blockSize=5,
    )
    if corners is not None and len(corners) >= int(cfg.min_matches):
        return corners.reshape(-1, 2).astype(np.float32)

    ys, xs = np.nonzero(query_mask)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > int(cfg.max_query_points):
        rng = np.random.default_rng(int(cfg.random_seed) + 101)
        ids = rng.choice(len(pts), int(cfg.max_query_points), replace=False)
        pts = pts[ids]
    return pts.astype(np.float32)


def _track_points_lk(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    query_xy: np.ndarray,
    cfg: TapirMatchConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if len(query_xy) == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.float32), {"backend": "opencv_lk", "reason": "no query points"}

    gray0 = _prepare_gray(before_rgb)
    gray1 = _prepare_gray(after_rgb)
    pts0 = np.asarray(query_xy, dtype=np.float32).reshape(-1, 1, 2)
    pts1, status, err = cv2.calcOpticalFlowPyrLK(
        gray0,
        gray1,
        pts0,
        None,
        winSize=(int(cfg.lk_win_size), int(cfg.lk_win_size)),
        maxLevel=int(cfg.lk_max_level),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if pts1 is None or status is None:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.float32), {"backend": "opencv_lk", "reason": "LK returned no tracks"}

    err_flat = np.zeros(len(pts0), dtype=np.float32) if err is None else err.reshape(-1).astype(np.float32)
    tracked0 = pts0.reshape(-1, 2)
    tracked1 = pts1.reshape(-1, 2).astype(np.float32)
    keep = status.reshape(-1).astype(bool)
    keep &= np.isfinite(tracked1).all(axis=1)
    keep &= err_flat <= float(cfg.max_lk_error)
    h, w = gray1.shape[:2]
    keep &= (
        (tracked1[:, 0] >= 0.0) & (tracked1[:, 0] < float(w)) &
        (tracked1[:, 1] >= 0.0) & (tracked1[:, 1] < float(h))
    )
    quality = 1.0 / (1.0 + np.maximum(err_flat[keep], 0.0))
    return tracked0[keep], tracked1[keep], quality.astype(np.float32), {
        "backend": "opencv_lk",
        "num_query_points": int(len(query_xy)),
        "num_tracked_points": int(keep.sum()),
        "mean_lk_error": float(err_flat[keep].mean()) if keep.any() else math.inf,
    }


def _track_points_tapir_torch(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    query_xy: np.ndarray,
    cfg: TapirMatchConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    torch, _, _ = _ensure_tapir_torch_imports()
    before = _as_uint8_rgb(before_rgb)
    after = _as_uint8_rgb(after_rgb)
    query_xy = np.asarray(query_xy, dtype=np.float32).reshape(-1, 2)
    if len(query_xy) == 0:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.float32), {
            "backend": "tapir_torch",
            "num_query_points": 0,
            "num_tracked_points": 0,
        }

    resize_h = int(getattr(cfg, "tapir_resize_height", 256) or 256)
    resize_w = int(getattr(cfg, "tapir_resize_width", 256) or 256)
    query_chunk_size = int(getattr(cfg, "tapir_query_chunk_size", 64) or 64)
    causal = bool(getattr(cfg, "tapir_causal", False))
    model, device_obj, resolved_ckpt = _load_tapir_model(
        str(cfg.tapir_checkpoint_path or ""),
        cfg.tapir_device,
        causal=causal,
    )

    h0, w0 = before.shape[:2]
    frame0 = cv2.resize(before, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    frame1 = cv2.resize(after, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    frames_np = np.stack([frame0, frame1], axis=0)

    query_scaled = query_xy.copy()
    query_scaled[:, 0] *= float(resize_w) / float(max(w0, 1))
    query_scaled[:, 1] *= float(resize_h) / float(max(h0, 1))
    query_tyx = np.empty((len(query_scaled), 3), dtype=np.float32)
    query_tyx[:, 0] = 0.0
    query_tyx[:, 1] = query_scaled[:, 1]
    query_tyx[:, 2] = query_scaled[:, 0]

    frames = torch.from_numpy(frames_np).to(device_obj)
    query = torch.from_numpy(query_tyx).to(device_obj)
    with torch.no_grad():
        outputs = model(
            _preprocess_tapir_frames(frames)[None],
            query[None].float(),
            query_chunk_size=query_chunk_size,
        )
    tracks = outputs["tracks"][0].detach()
    confidence = _tapir_track_confidence(
        outputs["occlusion"][0].detach(),
        outputs["expected_dist"][0].detach(),
    )
    after_xy = tracks[:, 1, :].detach().cpu().numpy().astype(np.float32)
    after_xy[:, 0] *= float(after.shape[1]) / float(resize_w)
    after_xy[:, 1] *= float(after.shape[0]) / float(resize_h)
    quality = confidence[:, 1].detach().cpu().numpy().astype(np.float32)
    keep = np.isfinite(after_xy).all(axis=1) & np.isfinite(quality)
    keep &= quality > 0.0
    debug = {
        "backend": "tapir_torch",
        "checkpoint_path": resolved_ckpt,
        "device": str(device_obj),
        "resize_hw": [resize_h, resize_w],
        "num_query_points": int(len(query_xy)),
        "num_tracked_points": int(keep.sum()),
        "mean_track_quality": float(quality[keep].mean()) if keep.any() else 0.0,
    }
    return (
        query_xy[keep].astype(np.float32),
        after_xy[keep].astype(np.float32),
        np.clip(quality[keep], 0.0, 1.0).astype(np.float32),
        debug,
    )


def _track_points(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    query_xy: np.ndarray,
    cfg: TapirMatchConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    backend = str(cfg.backend or "auto").lower()
    if backend not in {"auto", "tapir", "opencv_lk"}:
        raise ValueError(f"Unknown TAPIR backend: {cfg.backend}")
    if backend in {"auto", "tapir"}:
        try:
            return _track_points_tapir_torch(before_rgb, after_rgb, query_xy, cfg)
        except Exception as exc:
            if backend == "tapir":
                raise
            k0, k1, q, debug = _track_points_lk(before_rgb, after_rgb, query_xy, cfg)
            debug["tapir_torch_error"] = str(exc)
            debug["backend"] = "opencv_lk_fallback"
            return k0, k1, q, debug
    return _track_points_lk(before_rgb, after_rgb, query_xy, cfg)


def _estimate_transform_from_correspondences(
    before_rgb: np.ndarray,
    before_depth_m: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    after_depth_m: np.ndarray,
    intr: Any,
    cfg: TapirMatchConfig,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    track_quality: np.ndarray,
    debug: dict[str, Any],
    timings: dict[str, float],
) -> TapirInitResult:
    matches = np.column_stack([
        np.arange(len(kpts0), dtype=np.int32),
        np.arange(len(kpts1), dtype=np.int32),
    ]).astype(np.int32)
    debug["num_keypoints_before"] = int(len(kpts0))
    debug["num_keypoints_after"] = int(len(kpts1))
    debug["num_matches"] = int(len(matches))
    if len(matches) < int(cfg.min_matches):
        debug.update({"success": False, "reason": "too few raw tracks"})
        return TapirInitResult(False, before_keypoints=kpts0, after_keypoints=kpts1, matches=matches, debug=debug)

    t_stage = time.perf_counter()
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
    keep_region = (keep_mask | keep_roi_near) if cfg.use_roi_fallback else keep_mask
    keep = keep_region & valid0 & valid1
    match_weights = np.full(len(matches), 0.01, dtype=np.float64)
    match_weights[keep_roi] = 0.06
    match_weights[keep_roi_near] = 0.55
    match_weights[keep_mask] = 1.0
    if len(track_quality) == len(match_weights):
        match_weights *= np.clip(np.asarray(track_quality, dtype=np.float64), 0.05, 1.0)
    valid_match_indices = _select_valid_matches(
        keep,
        match_weights,
        cfg.max_valid_matches,
        np.random.default_rng(int(cfg.random_seed) + 117),
    )
    timings["filter_tracks"] = time.perf_counter() - t_stage

    debug.update({
        "num_before_mask_matches": int(keep_mask_before.sum()),
        "num_after_mask_matches": int(keep_mask_after.sum()),
        "num_mask_matches": int(keep_mask.sum()),
        "num_roi_matches": int(keep_roi.sum()),
        "num_near_mask_matches": int(keep_near_mask.sum()),
        "num_roi_near_mask_matches": int(keep_roi_near.sum()),
        "num_before_depth_matches": int(valid0.sum()),
        "num_after_depth_matches": int(valid1.sum()),
        "num_region_depth_matches": int(keep.sum()),
        "num_valid_3d_matches": int(len(valid_match_indices)),
    })
    if len(valid_match_indices) < int(cfg.min_matches):
        debug.update({"success": False, "reason": "too few mask/depth-valid tracks"})
        return TapirInitResult(
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
        T_relaxed, inliers_relaxed, relaxed_debug = _ransac_rigid_transform(src3, tgt3, relaxed_cfg, weights=valid_weights)
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
    debug["num_near_mask_inliers"] = int((keep_near_mask[inlier_match_indices]).sum()) if len(inlier_match_indices) else 0
    debug["near_mask_inlier_ratio"] = float(debug["num_near_mask_inliers"] / max(len(inlier_match_indices), 1))
    rmse = float(debug.get("ransac_rmse_m", math.inf))
    req_min_inliers = int(cfg.small_sample_min_inliers) if fallback_used else int(cfg.min_inliers)
    req_min_ratio = float(cfg.small_sample_min_inlier_ratio) if fallback_used else float(cfg.min_inlier_ratio)
    req_max_rmse = float(cfg.small_sample_max_ransac_rmse_m) if fallback_used else float(cfg.max_ransac_rmse_m)
    near_mask_ok = debug["near_mask_inlier_ratio"] >= float(cfg.min_near_mask_inlier_ratio)
    projection_ok = True
    if float(cfg.min_object_projection_near_mask_ratio) > 0.0 and T is not None:
        projection_debug = _object_projection_support(
            before_depth_m=before_depth_m,
            before_mask=before_mask,
            after_mask=after_mask,
            T_before_to_after=T,
            intr=intr,
            max_depth_m=cfg.max_depth_m,
            near_mask_px=cfg.roi_near_mask_max_dist_px,
            max_samples=cfg.object_projection_sample_count,
            random_seed=int(cfg.random_seed) + 129,
        )
        debug.update(projection_debug)
        projection_ok = (
            float(projection_debug.get("object_projection_near_mask_ratio", 0.0))
            >= float(cfg.min_object_projection_near_mask_ratio)
        )
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
            debug.setdefault("reason", "tapir transform rejected")
        T = None

    return TapirInitResult(
        success=bool(debug["success"]),
        T_before_cam_to_after_cam=T,
        before_keypoints=kpts0,
        after_keypoints=kpts1,
        matches=matches,
        valid_match_indices=valid_match_indices,
        inlier_match_indices=inlier_match_indices,
        debug=debug,
    )


def estimate_tapir_init_transform(
    before_rgb: np.ndarray,
    before_depth_m: np.ndarray,
    before_mask: np.ndarray,
    after_rgb: np.ndarray,
    after_depth_m: np.ndarray,
    after_mask: np.ndarray,
    intr: Any,
    cfg: TapirMatchConfig | None = None,
) -> TapirInitResult:
    cfg = cfg or TapirMatchConfig()
    debug: dict[str, Any] = {"enabled": True, "method": "tapir"}
    timings: dict[str, float] = {}
    t_total = time.perf_counter()
    try:
        t_stage = time.perf_counter()
        query_xy = _sample_query_points(before_rgb, before_depth_m, before_mask, cfg)
        timings["sample_queries"] = time.perf_counter() - t_stage

        t_stage = time.perf_counter()
        kpts0, kpts1, track_quality, track_debug = _track_points(before_rgb, after_rgb, query_xy, cfg)
        timings["track"] = time.perf_counter() - t_stage
        debug.update(track_debug)

        result = _estimate_transform_from_correspondences(
            before_rgb=before_rgb,
            before_depth_m=before_depth_m,
            before_mask=before_mask,
            after_mask=after_mask,
            after_depth_m=after_depth_m,
            intr=intr,
            cfg=cfg,
            kpts0=kpts0,
            kpts1=kpts1,
            track_quality=track_quality,
            debug=debug,
            timings=timings,
        )
        timings["total"] = time.perf_counter() - t_total
        result.debug["timings_s"] = timings
        return result
    except Exception as exc:
        timings["total"] = time.perf_counter() - t_total
        debug["timings_s"] = timings
        debug.update({"success": False, "error": str(exc), "stage": "tapir_track"})
        return TapirInitResult(success=False, debug=debug)


def draw_tapir_matches(before_rgb: np.ndarray, after_rgb: np.ndarray, result: TapirInitResult, max_lines: int = 180) -> np.ndarray:
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
        cv2.putText(canvas, "No TAPIR tracks", (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)
        return canvas

    def _match_color(match_idx: int, muted: bool = False) -> tuple[int, int, int]:
        hue = float((int(match_idx) * 37) % 180) / 180.0
        hsv = np.array([[[hue * 179.0, 210.0, 255.0]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0].astype(np.float32)
        if muted:
            rgb = 0.45 * rgb + 0.55 * np.array([120.0, 120.0, 120.0], dtype=np.float32)
        return tuple(int(v) for v in np.clip(rgb, 0, 255))

    inlier_set = set(int(i) for i in result.inlier_match_indices.tolist())
    draw_indices = result.valid_match_indices
    raw_mode = False
    if len(draw_indices) == 0:
        raw_mode = True
        draw_indices = np.arange(len(result.matches), dtype=np.int32)
    if len(draw_indices) > max_lines:
        rng = np.random.default_rng(11)
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
            f"{'multi-color raw tracks' if raw_mode else 'bright=inlier muted=outlier'}"
        ),
        (16, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas
