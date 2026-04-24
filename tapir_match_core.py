from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]

from match_geometry_core import (
    backproject as _backproject,
    estimate_rigid_transform as _estimate_rigid_transform_weighted,
    filter_projected_uv as _filter_projected_uv,
    mask_contains as _mask_contains,
    mask_distance_values as _mask_distance_values,
    object_projection_support as _object_projection_support,
    projected_uv_to_mask as _projected_uv_to_mask,
    ransac_rigid_transform as _ransac_rigid_transform,
    roi_contains as _roi_contains,
    sample_depth as _sample_depth,
    select_valid_matches as _select_valid_matches,
)


@dataclass
class TapirMatchConfig:
    min_matches: int = 4
    min_inliers: int = 4
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
    target_valid_matches: int = 36
    match_spatial_separation_px: float = 10.0
    random_seed: int = 0
    backend: str = "remote_tapnet"
    remote_host: str = "192.168.20.212"
    remote_port: int = 19812
    remote_protocol: str = "ws"
    remote_timeout_s: float = 60.0
    tapir_checkpoint_path: str | None = None
    tapir_device: str | None = None
    tapir_resize_height: int = 256
    tapir_resize_width: int = 256
    max_query_points: int = 100
    sampling_mode: str = "uniform_random"
    sampling_retry_enable: bool = True
    sampling_retry_modes: tuple[str, ...] = ("grid",)
    sampling_retry_min_inliers: int = 5
    sampling_retry_min_proj_iou: float = 0.38
    sampling_retry_max_rel_dense_rmse: float = 0.055
    sampling_retry_min_proj_precision: float = 0.55
    sampling_retry_failed_min_inliers: int = 5
    sampling_retry_failed_min_proj_iou: float = 0.32
    sampling_retry_failed_max_rel_dense_rmse: float = 0.065
    requested_points: int = 100
    prior_confidence_accept_thresh: float = 0.60
    prior_confidence_strong_thresh: float = 0.72
    dense_eval_max_points: int = 2048
    projection_eval_downsample: int = 4
    anchor_topk: int = 8
    anchor_weight_boost: float = 2.5
    anchor_refine_iters: int = 4
    min_track_quality: float = 0.01
    min_track_distance_px: int = 6
    verbose_timings: bool = False
    log_fn: Any = None
    case_tag: str = ""


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
_TAPIR_REMOTE_CLIENT_CLASS = None
_TAPIR_REMOTE_CLIENT_IMPORT_ERROR: Exception | None = None


def _emit_timing_log(cfg: TapirMatchConfig | None, message: str) -> None:
    if cfg is None or not bool(getattr(cfg, "verbose_timings", False)):
        return
    prefix = str(getattr(cfg, "case_tag", "") or "").strip()
    stamp = time.strftime("%H:%M:%S")
    text = f"[{stamp}]"
    if prefix:
        text += f" [{prefix}]"
    text += f" {message}"
    log_fn = getattr(cfg, "log_fn", None)
    if callable(log_fn):
        log_fn(text)
    else:
        print(text, flush=True)


def _ensure_tapir_remote_client_class():
    global _TAPIR_REMOTE_CLIENT_CLASS, _TAPIR_REMOTE_CLIENT_IMPORT_ERROR
    if _TAPIR_REMOTE_CLIENT_CLASS is not None:
        return _TAPIR_REMOTE_CLIENT_CLASS
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    try:
        from tapir.remote import RemoteTrackerClient
    except Exception as exc:
        _TAPIR_REMOTE_CLIENT_IMPORT_ERROR = exc
        raise ImportError(f"TAPIR remote client import failed: {exc}") from exc
    _TAPIR_REMOTE_CLIENT_CLASS = RemoteTrackerClient
    _TAPIR_REMOTE_CLIENT_IMPORT_ERROR = None
    return _TAPIR_REMOTE_CLIENT_CLASS


def _as_uint8_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected RGB image HxWx3, got {arr.shape}")
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _prepare_gray(image_rgb: np.ndarray) -> np.ndarray:
    arr = np.asarray(image_rgb)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int]:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    ys, xs = np.nonzero(m > 0)
    if len(xs) == 0:
        return [0, 0, -1, -1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _sample_query_points(
    image_rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    cfg: TapirMatchConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    gray = _prepare_gray(image_rgb)
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m > 0).astype(np.uint8)
    valid_depth = np.isfinite(depth_m) & (depth_m > 1e-5) & (depth_m < float(cfg.max_depth_m))
    query_mask = (m > 0) & valid_depth
    requested_points = max(1, int(getattr(cfg, "requested_points", 0) or int(cfg.max_query_points)))
    sampling_mode = str(getattr(cfg, "sampling_mode", "uniform_random") or "uniform_random").strip().lower()
    debug = {
        "sampling_mode": sampling_mode,
        "requested_points": int(requested_points),
        "random_seed": int(cfg.random_seed),
        "source_crop_bbox_xyxy": _mask_bbox_xyxy(query_mask),
    }
    if not query_mask.any():
        debug["sampled_points_xy"] = []
        return np.zeros((0, 2), dtype=np.float32), debug

    def _sample_grid_points(mask_bool: np.ndarray, max_points: int) -> np.ndarray:
        ys_all, xs_all = np.nonzero(mask_bool)
        if len(xs_all) == 0 or max_points <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        pts_all = np.column_stack([xs_all, ys_all]).astype(np.float32)
        if len(pts_all) <= max_points:
            return pts_all

        x0, y0, x1, y1 = _mask_bbox_xyxy(mask_bool)
        width = max(1, int(x1 - x0 + 1))
        height = max(1, int(y1 - y0 + 1))
        aspect = float(width / max(height, 1))
        grid_cols = max(1, int(round(math.sqrt(max_points * max(aspect, 1e-6)))))
        grid_rows = max(1, int(math.ceil(max_points / max(grid_cols, 1))))
        picked: list[tuple[float, float]] = []
        used: set[tuple[int, int]] = set()
        for row in range(grid_rows):
            y_lo = y0 + int(math.floor(row * height / grid_rows))
            y_hi = y0 + int(math.floor((row + 1) * height / grid_rows))
            if row == grid_rows - 1:
                y_hi = y1 + 1
            for col in range(grid_cols):
                x_lo = x0 + int(math.floor(col * width / grid_cols))
                x_hi = x0 + int(math.floor((col + 1) * width / grid_cols))
                if col == grid_cols - 1:
                    x_hi = x1 + 1
                cell = mask_bool[y_lo:y_hi, x_lo:x_hi]
                if not np.any(cell):
                    continue
                ys_cell, xs_cell = np.nonzero(cell)
                xs_abs = xs_cell + x_lo
                ys_abs = ys_cell + y_lo
                cx = 0.5 * float(x_lo + x_hi - 1)
                cy = 0.5 * float(y_lo + y_hi - 1)
                d2 = np.square(xs_abs.astype(np.float32) - cx) + np.square(ys_abs.astype(np.float32) - cy)
                idx = int(np.argmin(d2))
                key = (int(xs_abs[idx]), int(ys_abs[idx]))
                if key in used:
                    continue
                used.add(key)
                picked.append((float(key[0]), float(key[1])))
                if len(picked) >= max_points:
                    break
            if len(picked) >= max_points:
                break
        if len(picked) < max_points:
            step_ids = np.linspace(0, len(pts_all) - 1, num=max_points, dtype=np.int32)
            for idx in step_ids.tolist():
                key = (int(pts_all[idx, 0]), int(pts_all[idx, 1]))
                if key in used:
                    continue
                used.add(key)
                picked.append((float(key[0]), float(key[1])))
                if len(picked) >= max_points:
                    break
        return np.asarray(picked[:max_points], dtype=np.float32)

    if sampling_mode in {"good_features", "gftt", "corners"}:
        corners = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=int(requested_points),
            qualityLevel=float(cfg.min_track_quality),
            minDistance=float(cfg.min_track_distance_px),
            mask=(query_mask.astype(np.uint8) * 255),
            blockSize=5,
        )
        if corners is not None and len(corners) >= int(cfg.min_matches):
            pts = corners.reshape(-1, 2).astype(np.float32)
            debug["sampled_points_xy"] = np.round(pts).astype(int).tolist()
            return pts, debug
    elif sampling_mode in {"grid", "mask_grid", "stratified"}:
        pts = _sample_grid_points(query_mask, int(requested_points))
        debug["sampled_points_xy"] = np.round(pts).astype(int).tolist()
        return pts, debug

    ys, xs = np.nonzero(query_mask)
    if len(xs) == 0:
        debug["sampled_points_xy"] = []
        return np.zeros((0, 2), dtype=np.float32), debug
    pts = np.column_stack([xs, ys]).astype(np.float32)
    if len(pts) > int(requested_points):
        rng = np.random.default_rng(int(cfg.random_seed))
        ids = rng.choice(len(pts), int(requested_points), replace=False)
        pts = pts[ids]
    pts = pts.astype(np.float32)
    debug["sampled_points_xy"] = np.round(pts).astype(int).tolist()
    return pts, debug


def _remote_debug_base(cfg: TapirMatchConfig, num_query_points: int) -> dict[str, Any]:
    return {
        "backend": "remote_tapnet",
        "remote_host": str(getattr(cfg, "remote_host", "") or "192.168.20.212"),
        "remote_port": int(getattr(cfg, "remote_port", 19812) or 19812),
        "remote_protocol": str(getattr(cfg, "remote_protocol", "ws") or "ws"),
        "num_query_points": int(num_query_points),
    }


def _tapir_retry_needed(debug: dict[str, Any], cfg: TapirMatchConfig) -> tuple[bool, list[str]]:
    if not bool(getattr(cfg, "sampling_retry_enable", True)):
        return False, []

    inliers = int(debug.get("num_inliers", debug.get("inlier_count", 0)) or 0)
    valid_matches = int(debug.get("num_valid_3d_matches", debug.get("valid_3d_pairs", 0)) or 0)
    proj_iou = float(debug.get("prior_proj_iou", 0.0) or 0.0)
    proj_precision = float(debug.get("prior_proj_precision", 0.0) or 0.0)
    rel_dense_rmse = float(debug.get("prior_dense_relative_rmse_eval", math.inf) or math.inf)
    max_inliers = int(getattr(cfg, "sampling_retry_min_inliers", 5) or 5)
    min_proj_iou = float(getattr(cfg, "sampling_retry_min_proj_iou", 0.38) or 0.38)
    max_rel_dense_rmse = float(getattr(cfg, "sampling_retry_max_rel_dense_rmse", 0.055) or 0.055)
    min_proj_precision = float(getattr(cfg, "sampling_retry_min_proj_precision", 0.55) or 0.55)
    failed_min_inliers = int(getattr(cfg, "sampling_retry_failed_min_inliers", 5) or 5)
    failed_min_proj_iou = float(getattr(cfg, "sampling_retry_failed_min_proj_iou", 0.32) or 0.32)
    failed_max_rel_dense_rmse = float(getattr(cfg, "sampling_retry_failed_max_rel_dense_rmse", 0.065) or 0.065)

    def _promising_near_miss(
        min_inliers_needed: int,
        min_proj_iou_needed: float,
        max_rel_dense_rmse_allowed: float,
    ) -> bool:
        enough_signal = max(inliers, valid_matches) >= int(min_inliers_needed)
        proj_ok = proj_iou >= float(min_proj_iou_needed)
        precision_ok = proj_precision >= float(min_proj_precision)
        dense_ok = np.isfinite(rel_dense_rmse) and rel_dense_rmse <= float(max_rel_dense_rmse_allowed)
        return bool(enough_signal and proj_ok and precision_ok and dense_ok)

    if not bool(debug.get("success", False)):
        reason = str(debug.get("reason", "") or "")
        if reason in {"too few raw tracks", "too few mask/depth-valid tracks"}:
            return False, []
        if _promising_near_miss(
            min_inliers_needed=failed_min_inliers,
            min_proj_iou_needed=failed_min_proj_iou,
            max_rel_dense_rmse_allowed=failed_max_rel_dense_rmse,
        ):
            reasons = ["failed"]
            if inliers < failed_min_inliers:
                reasons.append("low_inliers")
            if proj_iou < min_proj_iou:
                reasons.append("low_prior_proj_iou")
            if rel_dense_rmse > max_rel_dense_rmse:
                reasons.append("high_rel_dense_rmse")
            return True, reasons
        return False, []

    low_inliers = inliers <= max_inliers
    low_proj_iou = proj_iou < min_proj_iou
    high_rel_dense_rmse = rel_dense_rmse > max_rel_dense_rmse
    if low_inliers and (low_proj_iou or high_rel_dense_rmse) and _promising_near_miss(
        min_inliers_needed=max(int(cfg.min_inliers), 4),
        min_proj_iou_needed=max(0.30, min_proj_iou - 0.03),
        max_rel_dense_rmse_allowed=max_rel_dense_rmse,
    ):
        reasons: list[str] = ["low_inliers"]
        if low_proj_iou:
            reasons.append("low_prior_proj_iou")
        if high_rel_dense_rmse:
            reasons.append("high_rel_dense_rmse")
        return True, reasons
    return False, []


def _prior_confidence_from_debug(debug: dict[str, Any], cfg: TapirMatchConfig) -> tuple[float, dict[str, float]]:
    proj_iou = float(debug.get("prior_proj_iou", 0.0) or 0.0)
    rel_dense_rmse = float(debug.get("prior_dense_relative_rmse_eval", math.inf) or math.inf)
    ransac_rmse = float(debug.get("ransac_rmse_m", debug.get("rmse_m", math.inf)) or math.inf)
    inliers = float(debug.get("num_inliers", debug.get("inlier_count", 0)) or 0.0)

    proj_score = float(np.clip((proj_iou - 0.10) / 0.70, 0.0, 1.0))
    dense_score = 0.0 if not np.isfinite(rel_dense_rmse) else float(np.clip((0.12 - rel_dense_rmse) / 0.12, 0.0, 1.0))
    ransac_score = 0.0 if not np.isfinite(ransac_rmse) else float(np.clip((0.018 - ransac_rmse) / 0.018, 0.0, 1.0))
    inlier_floor = max(float(cfg.min_inliers), 4.0)
    inlier_score = float(np.clip((inliers - inlier_floor) / 20.0, 0.0, 1.0))
    confidence = float(
        0.40 * proj_score
        + 0.30 * dense_score
        + 0.20 * ransac_score
        + 0.10 * inlier_score
    )
    return confidence, {
        "prior_confidence_proj_score": float(proj_score),
        "prior_confidence_dense_score": float(dense_score),
        "prior_confidence_ransac_score": float(ransac_score),
        "prior_confidence_inlier_score": float(inlier_score),
    }


def _prior_accept_override(debug: dict[str, Any], cfg: TapirMatchConfig) -> tuple[bool, dict[str, float | bool | str]]:
    proj_iou = float(debug.get("prior_proj_iou", 0.0) or 0.0)
    rel_dense_rmse = float(debug.get("prior_dense_relative_rmse_eval", math.inf) or math.inf)
    ransac_rmse = float(debug.get("ransac_rmse_m", debug.get("rmse_m", math.inf)) or math.inf)
    inliers = int(debug.get("num_inliers", debug.get("inlier_count", 0)) or 0)
    proj_precision = float(debug.get("prior_proj_precision", 0.0) or 0.0)

    min_inliers = max(int(getattr(cfg, "min_inliers", 4) or 4), 5)
    override = bool(
        inliers >= min_inliers
        and proj_iou >= 0.44
        and proj_precision >= 0.58
        and np.isfinite(rel_dense_rmse)
        and rel_dense_rmse <= 0.050
        and np.isfinite(ransac_rmse)
        and ransac_rmse <= min(float(cfg.max_ransac_rmse_m), 0.0145)
    )
    return override, {
        "prior_gate_borderline_override": bool(override),
        "prior_gate_borderline_proj_iou_thresh": 0.44,
        "prior_gate_borderline_proj_precision_thresh": 0.58,
        "prior_gate_borderline_rel_dense_rmse_thresh": 0.050,
        "prior_gate_borderline_ransac_rmse_thresh": float(min(float(cfg.max_ransac_rmse_m), 0.0145)),
        "prior_gate_borderline_min_inliers": int(min_inliers),
        "prior_gate_borderline_reason": ("borderline_projection_dense_override" if override else ""),
    }


def _tapir_result_rank(result: TapirInitResult) -> tuple[float, ...]:
    debug = dict(result.debug or {})
    success = 1.0 if bool(debug.get("success", result.success)) else 0.0
    prior_confidence = float(debug.get("prior_confidence", 0.0) or 0.0)
    inliers = float(debug.get("num_inliers", debug.get("inlier_count", 0)) or 0.0)
    proj_iou = float(debug.get("prior_proj_iou", 0.0) or 0.0)
    proj_precision = float(debug.get("prior_proj_precision", 0.0) or 0.0)
    rel_dense_rmse = float(debug.get("prior_dense_relative_rmse_eval", math.inf) or math.inf)
    ransac_rmse = float(debug.get("ransac_rmse_m", debug.get("rmse_m", math.inf)) or math.inf)
    if not np.isfinite(rel_dense_rmse):
        rel_dense_rmse = 1e6
    if not np.isfinite(ransac_rmse):
        ransac_rmse = 1e6
    return (
        success,
        prior_confidence,
        proj_iou,
        proj_precision,
        inliers,
        -rel_dense_rmse,
        -ransac_rmse,
    )


def _track_points_remote_tapir_rpc(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    query_xy: np.ndarray,
    cfg: TapirMatchConfig,
    base_debug: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    RemoteTrackerClient = _ensure_tapir_remote_client_class()
    protocol = str(getattr(cfg, "remote_protocol", "ws") or "ws").strip().lower()
    if protocol not in {"auto", "ws", "websocket", "ws_mwrpc"}:
        raise ValueError(f"Unsupported TAPIR remote protocol: {cfg.remote_protocol}")
    host = str(getattr(cfg, "remote_host", "") or "192.168.20.212").strip()
    port = int(getattr(cfg, "remote_port", 19812) or 19812)
    if "://" in host:
        scheme, rest = host.split("://", 1)
        if scheme not in {"ws", "http"}:
            raise ValueError(f"Unsupported TAPIR RPC host scheme: {scheme}")
        host = rest.split("/", 1)[0]
    if ":" in host:
        host_only, maybe_port = host.rsplit(":", 1)
        if maybe_port.isdigit():
            host = host_only
            port = int(maybe_port)

    before_bgr = _as_uint8_bgr(before_rgb)
    after_bgr = _as_uint8_bgr(after_rgb)
    points = [(int(round(float(x))), int(round(float(y)))) for x, y in query_xy]
    client = None
    rpc_timings: dict[str, float] = {}
    rpc_client_debug: dict[str, dict[str, Any]] = {}
    rpc_server_debug: dict[str, dict[str, float]] = {}
    _emit_timing_log(cfg, f"TAPIR RPC start host={host}:{port} protocol={protocol} queries={len(points)}")
    try:
        t_rpc = time.perf_counter()
        client = RemoteTrackerClient(host=host, port=port)
        rpc_timings["construct"] = time.perf_counter() - t_rpc
        _emit_timing_log(cfg, f"TAPIR RPC client ready in {rpc_timings['construct'] * 1000.0:.1f} ms")

        t_rpc = time.perf_counter()
        _emit_timing_log(cfg, "TAPIR RPC init() ...")
        client.init()
        rpc_timings["init"] = time.perf_counter() - t_rpc
        rpc_client_debug["init"] = dict(getattr(client, "last_init_debug", {}) or {})
        rpc_server_debug["init"] = {
            key: float(value)
            for key, value in (rpc_client_debug["init"].get("server_timings_s", {}) or {}).items()
        }
        _emit_timing_log(cfg, f"TAPIR RPC init() done in {rpc_timings['init'] * 1000.0:.1f} ms")

        t_rpc = time.perf_counter()
        _emit_timing_log(cfg, f"TAPIR RPC add_points() ... count={len(points)}")
        client.add_points(before_bgr, points)
        rpc_timings["add_points"] = time.perf_counter() - t_rpc
        rpc_client_debug["add_points"] = dict(getattr(client, "last_add_points_debug", {}) or {})
        rpc_server_debug["add_points"] = {
            key: float(value)
            for key, value in (rpc_client_debug["add_points"].get("server_timings_s", {}) or {}).items()
        }
        _emit_timing_log(cfg, f"TAPIR RPC add_points() done in {rpc_timings['add_points'] * 1000.0:.1f} ms")

        t_rpc = time.perf_counter()
        _emit_timing_log(cfg, "TAPIR RPC predict() ...")
        tracks, visible = client.predict(after_bgr)
        rpc_timings["predict"] = time.perf_counter() - t_rpc
        rpc_client_debug["predict"] = dict(getattr(client, "last_predict_debug", {}) or {})
        rpc_server_debug["predict"] = {
            key: float(value)
            for key, value in (rpc_client_debug["predict"].get("server_timings_s", {}) or {}).items()
        }
        _emit_timing_log(cfg, f"TAPIR RPC predict() done in {rpc_timings['predict'] * 1000.0:.1f} ms")
    except Exception as exc:
        _emit_timing_log(cfg, f"TAPIR RPC failed: {exc}")
        raise
    finally:
        if client is not None:
            close = getattr(client.client, "close", None)
            if callable(close):
                try:
                    t_rpc = time.perf_counter()
                    close()
                    rpc_timings["close"] = time.perf_counter() - t_rpc
                    _emit_timing_log(cfg, f"TAPIR RPC close() done in {rpc_timings['close'] * 1000.0:.1f} ms")
                except Exception:
                    pass

    tracks = np.asarray(tracks, dtype=np.float32).reshape(-1, 2)
    visible = np.asarray(visible).reshape(-1).astype(bool)
    n = min(len(query_xy), len(tracks), len(visible))
    query_xy = query_xy[:n]
    tracks = tracks[:n]
    visible = visible[:n]
    quality = visible.astype(np.float32)
    h, w = after_bgr.shape[:2]
    keep = visible & np.isfinite(tracks).all(axis=1)
    keep &= (
        (tracks[:, 0] >= 0.0) & (tracks[:, 0] < float(w)) &
        (tracks[:, 1] >= 0.0) & (tracks[:, 1] < float(h))
    )
    debug = dict(base_debug or {})
    client_encode_total_s = float(sum(
        float((rpc_client_debug.get(name, {}) or {}).get("encode_s", 0.0))
        for name in ("add_points", "predict")
    ))
    server_decode_total_s = float(
        float((rpc_server_debug.get("add_points", {}) or {}).get("decode", 0.0))
        + float((rpc_server_debug.get("predict", {}) or {}).get("decode", 0.0))
    )
    server_compute_total_s = float(
        float((rpc_server_debug.get("init", {}) or {}).get("tracker_init", 0.0))
        + float((rpc_server_debug.get("add_points", {}) or {}).get("tracker_add_points", 0.0))
        + float((rpc_server_debug.get("predict", {}) or {}).get("tracker_predict", 0.0))
    )
    rpc_total_s = float(sum(float(v) for v in rpc_timings.values()))
    rpc_io_overhead_s = max(0.0, rpc_total_s - server_compute_total_s)
    debug.update({
        "backend": "remote_tapnet",
        "remote_protocol": "ws_mwrpc",
        "remote_endpoint": "mwrpc",
        "remote_mode": "stateful_rpc",
        "num_query_points": int(len(points)),
        "num_tracked_points": int(keep.sum()),
        "mean_track_quality": float(np.mean(quality[keep])) if keep.any() else 0.0,
        "tapir_package": "tapir.remote.RemoteTrackerClient",
        "tapir_latency_ms": None,
        "tapir_latency_no_rpc_io_ms": int(round(server_compute_total_s * 1000.0)),
        "tapir_latency_no_remote_roundtrip_ms": None,
        "visible_count": int(keep.sum()),
        "visible_ratio": float(float(keep.sum()) / max(len(points), 1)),
        "tapir": {
            "endpoint": "tapir_remote",
            "latency_ms": None,
            "latency_no_rpc_io_ms": int(round(server_compute_total_s * 1000.0)),
            "latency_no_remote_roundtrip_ms": None,
            "visible_count": int(keep.sum()),
            "visible_ratio": float(float(keep.sum()) / max(len(points), 1)),
            "host": host,
            "port": port,
        },
        "rpc_timings_s": {key: float(value) for key, value in rpc_timings.items()},
        "rpc_client_debug": rpc_client_debug,
        "rpc_server_timings_s": rpc_server_debug,
        "tapir_rpc_client_encode_ms": int(round(client_encode_total_s * 1000.0)),
        "tapir_rpc_server_decode_ms": int(round(server_decode_total_s * 1000.0)),
        "tapir_rpc_server_compute_ms": int(round(server_compute_total_s * 1000.0)),
        "tapir_rpc_io_overhead_ms": int(round(rpc_io_overhead_s * 1000.0)),
    })
    if rpc_timings:
        debug["tapir_rpc_total_ms"] = int(round(rpc_total_s * 1000.0))
        _emit_timing_log(
            cfg,
            "TAPIR RPC summary "
            + " ".join(f"{key}={float(value) * 1000.0:.1f}ms" for key, value in rpc_timings.items()),
        )
    return query_xy[keep], tracks[keep], quality[keep], debug


def _surface_points_from_mask_depth(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intr: Any,
    max_depth_m: float,
    max_points: int,
    random_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    valid = (m > 0) & np.isfinite(depth_m) & (depth_m > 1e-5) & (depth_m < float(max_depth_m))
    ys, xs = np.nonzero(valid)
    total = int(len(xs))
    if total == 0:
        return np.zeros((0, 3), dtype=np.float64), {
            "input_points": 0,
            "used_points": 0,
            "max_points": int(max_points),
            "applied": False,
            "strategy": "empty",
            "voxel_size_m": 0.0,
            "representative_points": 0,
        }
    if int(max_points) > 0 and total > int(max_points):
        rng = np.random.default_rng(int(random_seed))
        sel = rng.choice(total, int(max_points), replace=False)
        xs = xs[sel]
        ys = ys[sel]
    xy = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    z = depth_m[ys, xs].astype(np.float64)
    pts = _backproject(xy, z, intr)
    return pts, {
        "input_points": total,
        "used_points": int(len(pts)),
        "max_points": int(max_points),
        "applied": bool(int(max_points) > 0 and total > int(max_points)),
        "strategy": "uniform_random" if int(max_points) > 0 else "disabled",
        "voxel_size_m": 0.0,
        "representative_points": int(len(pts)),
    }


def _select_diverse_match_indices(
    candidate_indices: np.ndarray,
    weights: np.ndarray,
    xy_target: np.ndarray,
    max_count: int,
    min_dist_px: float,
) -> np.ndarray:
    candidate_idx = np.asarray(candidate_indices, dtype=np.int32).reshape(-1)
    if len(candidate_idx) <= max_count:
        return np.sort(candidate_idx.astype(np.int32))
    target_xy = np.asarray(xy_target, dtype=np.float64).reshape(-1, 2)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    scores = weight_arr[candidate_idx]
    order = np.argsort(-scores, kind="stable")
    radius2 = max(float(min_dist_px), 0.0) ** 2
    selected: list[int] = []
    selected_pts: list[np.ndarray] = []
    for order_idx in order:
        cand = int(candidate_idx[order_idx])
        pt = target_xy[cand]
        if not np.isfinite(pt).all():
            continue
        if radius2 > 0.0 and selected_pts:
            d2 = np.sum((np.asarray(selected_pts, dtype=np.float64) - pt.reshape(1, 2)) ** 2, axis=1)
            if float(np.min(d2)) < radius2:
                continue
        selected.append(cand)
        selected_pts.append(pt.copy())
        if len(selected) >= int(max_count):
            break
    if len(selected) < int(max_count):
        selected_set = set(selected)
        for order_idx in order:
            cand = int(candidate_idx[order_idx])
            if cand in selected_set:
                continue
            selected.append(cand)
            selected_set.add(cand)
            if len(selected) >= int(max_count):
                break
    return np.sort(np.asarray(selected[: max_count], dtype=np.int32))


def _apply_transform(points: np.ndarray, tform: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(tform, dtype=np.float64).reshape(4, 4)
    return (pts @ T[:3, :3].T) + T[:3, 3]


def _anchor_local_refine(
    src3: np.ndarray,
    tgt3: np.ndarray,
    weights: np.ndarray,
    T_init: np.ndarray | None,
    inliers_mask: np.ndarray,
    cfg: TapirMatchConfig,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    active_idx = np.flatnonzero(np.asarray(inliers_mask, dtype=bool).reshape(-1))
    if T_init is None or len(active_idx) < 4:
        return T_init, np.asarray(inliers_mask, dtype=bool).reshape(-1), {
            "anchor_topk": int(getattr(cfg, "anchor_topk", 8)),
            "anchor_weight_boost": float(getattr(cfg, "anchor_weight_boost", 2.5)),
            "anchor_active_count_last_iter": int(len(active_idx)),
            "anchor_active_count_max": int(len(active_idx)),
            "anchor_rmse_last_m": math.inf,
            "anchor_rmse_best_m": math.inf,
            "anchor_refine_applied": False,
        }

    src_pts = np.asarray(src3, dtype=np.float64).reshape(-1, 3)
    tgt_pts = np.asarray(tgt3, dtype=np.float64).reshape(-1, 3)
    base_weights = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-8, None)
    topk = max(1, min(int(getattr(cfg, "anchor_topk", 8)), len(active_idx)))
    boost = max(1.0, float(getattr(cfg, "anchor_weight_boost", 2.5)))
    max_iters = max(1, int(getattr(cfg, "anchor_refine_iters", 4)))
    thresh = max(float(cfg.ransac_thresh_m), 1e-6)

    best_T = np.asarray(T_init, dtype=np.float64).reshape(4, 4)
    full_err = np.linalg.norm(_apply_transform(src_pts, best_T) - tgt_pts, axis=1)
    best_inliers = full_err <= thresh
    if int(best_inliers.sum()) < 4:
        best_inliers = np.asarray(inliers_mask, dtype=bool).reshape(-1)
    active_idx = np.flatnonzero(best_inliers)
    if len(active_idx) < 4:
        active_idx = np.flatnonzero(np.asarray(inliers_mask, dtype=bool).reshape(-1))
    active_count_max = int(len(active_idx))
    best_rmse = float(np.sqrt(np.mean(np.square(full_err[active_idx])))) if len(active_idx) else math.inf
    last_rmse = best_rmse
    applied = False

    for _ in range(max_iters):
        if len(active_idx) < 4:
            break
        active_err = np.linalg.norm(_apply_transform(src_pts[active_idx], best_T) - tgt_pts[active_idx], axis=1)
        order = np.argsort(active_err)
        anchor_local_idx = order[:topk]
        iter_weights = base_weights[active_idx].copy()
        iter_weights[anchor_local_idx] *= boost
        refined = _estimate_rigid_transform_weighted(src_pts[active_idx], tgt_pts[active_idx], iter_weights)
        if refined is None:
            break
        full_err = np.linalg.norm(_apply_transform(src_pts, refined) - tgt_pts, axis=1)
        refined_inliers = full_err <= thresh
        if int(refined_inliers.sum()) < 4:
            refined_inliers = np.asarray(inliers_mask, dtype=bool).reshape(-1)
        refined_active = np.flatnonzero(refined_inliers)
        if len(refined_active) < 4:
            break
        refined_rmse = float(np.sqrt(np.mean(np.square(full_err[refined_active]))))
        last_rmse = refined_rmse
        active_count_max = max(active_count_max, int(len(refined_active)))
        if refined_rmse <= best_rmse + 1e-9:
            best_T = refined
            best_rmse = refined_rmse
            best_inliers = refined_inliers
            active_idx = refined_active
            applied = True
        else:
            break

    return best_T, best_inliers, {
        "anchor_topk": int(topk),
        "anchor_weight_boost": float(boost),
        "anchor_active_count_last_iter": int(len(active_idx)),
        "anchor_active_count_max": int(active_count_max),
        "anchor_rmse_last_m": float(last_rmse),
        "anchor_rmse_best_m": float(best_rmse),
        "anchor_refine_applied": bool(applied),
    }


def _dense_transform_rmse(
    src_points: np.ndarray,
    tgt_points: np.ndarray,
    T_src_to_tgt: np.ndarray,
    inlier_thresh_m: float,
) -> tuple[float, float]:
    if len(src_points) == 0 or len(tgt_points) == 0:
        return math.inf, 0.0
    src_t = (np.asarray(src_points, dtype=np.float64) @ T_src_to_tgt[:3, :3].T) + T_src_to_tgt[:3, 3]
    tgt = np.asarray(tgt_points, dtype=np.float64)
    min_d2 = np.empty(len(src_t), dtype=np.float64)
    if o3d is not None and len(tgt):
        tgt_pcd = o3d.geometry.PointCloud()
        tgt_pcd.points = o3d.utility.Vector3dVector(tgt)
        tree = o3d.geometry.KDTreeFlann(tgt_pcd)
        for idx, point in enumerate(src_t):
            _, _, dist2 = tree.search_knn_vector_3d(point.astype(np.float64), 1)
            min_d2[idx] = float(dist2[0]) if dist2 else math.inf
    else:
        chunk = 512
        for i in range(0, len(src_t), chunk):
            part = src_t[i:i + chunk]
            d2 = np.sum((part[:, None, :] - tgt[None, :, :]) ** 2, axis=2)
            min_d2[i:i + len(part)] = np.min(d2, axis=1)
    rmse = float(np.sqrt(np.mean(min_d2))) if len(min_d2) else math.inf
    fitness = float(np.mean(min_d2 <= float(inlier_thresh_m) ** 2)) if len(min_d2) else 0.0
    return rmse, fitness


def _projected_surface_metrics(
    src_points: np.ndarray,
    T_src_to_tgt: np.ndarray,
    target_mask: np.ndarray,
    intr: Any,
    max_depth_m: float,
    downsample: int,
) -> dict[str, Any]:
    if len(src_points) == 0:
        return {"prior_proj_precision": 0.0, "prior_proj_recall": 0.0, "prior_proj_iou": 0.0, "prior_proj_pixels": 0}
    pts_t = (np.asarray(src_points, dtype=np.float64) @ T_src_to_tgt[:3, :3].T) + T_src_to_tgt[:3, 3]
    z = pts_t[:, 2]
    h, w = target_mask.shape[:2]
    front = np.isfinite(pts_t).all(axis=1) & (z > 1e-5) & (z < float(max_depth_m))
    if not np.any(front):
        return {"prior_proj_precision": 0.0, "prior_proj_recall": 0.0, "prior_proj_iou": 0.0, "prior_proj_pixels": 0}
    pts_f = pts_t[front]
    u = np.round((pts_f[:, 0] * float(intr.fx) / pts_f[:, 2]) + float(intr.cx)).astype(np.int32)
    v = np.round((pts_f[:, 1] * float(intr.fy) / pts_f[:, 2]) + float(intr.cy)).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        return {"prior_proj_precision": 0.0, "prior_proj_recall": 0.0, "prior_proj_iou": 0.0, "prior_proj_pixels": 0}
    ds = max(1, int(downsample))
    hs = max(1, (h + ds - 1) // ds)
    ws = max(1, (w + ds - 1) // ds)
    uv = np.column_stack([u[inside] // ds, v[inside] // ds]).astype(np.int32)
    uv[:, 0] = np.clip(uv[:, 0], 0, ws - 1)
    uv[:, 1] = np.clip(uv[:, 1], 0, hs - 1)
    proj_mask, uv_filtered = _projected_uv_to_mask(uv, (hs, ws), point_radius_px=1, connect_radius_px=2)
    if len(uv_filtered) == 0:
        return {"prior_proj_precision": 0.0, "prior_proj_recall": 0.0, "prior_proj_iou": 0.0, "prior_proj_pixels": 0}
    target_small = cv2.resize(((np.asarray(target_mask) > 0).astype(np.uint8) * 255), (ws, hs), interpolation=cv2.INTER_NEAREST)
    pred = proj_mask > 0
    gt = target_small > 0
    inter = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    gt_area = int(gt.sum())
    union = int(np.logical_or(pred, gt).sum())
    precision = float(inter / max(pred_area, 1))
    recall = float(inter / max(gt_area, 1))
    iou = float(inter / max(union, 1))
    return {
        "prior_proj_precision": precision,
        "prior_proj_recall": recall,
        "prior_proj_iou": iou,
        "prior_proj_pixels": pred_area,
    }


def _track_points(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    query_xy: np.ndarray,
    cfg: TapirMatchConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    backend = str(cfg.backend or "remote_tapnet").lower()
    if backend not in {"remote_tapnet", "remote", "tapir"}:
        raise ValueError(f"Unknown TAPIR backend: {cfg.backend}")
    return _track_points_remote_tapir_rpc(before_rgb, after_rgb, query_xy, cfg, _remote_debug_base(cfg, len(query_xy)))


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
    debug["target_crop_bbox_xyxy"] = _mask_bbox_xyxy(after_mask)
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
    match_weights[keep_roi] = 0.08
    match_weights[keep_mask] = 0.35
    match_weights[keep_near_mask] = np.maximum(match_weights[keep_near_mask], 0.80)
    match_weights[keep_roi_near] = 1.25
    if len(track_quality) == len(match_weights):
        match_weights *= np.clip(np.asarray(track_quality, dtype=np.float64), 0.05, 1.0)
    valid_candidate_count = int(keep.sum())
    max_valid_matches = min(
        int(cfg.max_valid_matches),
        max(1, int(getattr(cfg, "requested_points", 0) or int(cfg.max_query_points))),
        max(int(cfg.min_matches), int(getattr(cfg, "target_valid_matches", 36) or 36)),
    )
    min_dist_px = float(getattr(cfg, "match_spatial_separation_px", 10.0) or 0.0)
    preferred_keep = keep_roi_near & valid0 & valid1
    preferred_quota = min(max_valid_matches, max(int(cfg.min_matches) * 4, int(round(max_valid_matches * 0.75))))
    preferred_min = max(int(cfg.min_matches) * 2, min(12, max_valid_matches))
    preferred_indices = np.zeros(0, dtype=np.int32)
    if int(preferred_keep.sum()) >= preferred_min:
        preferred_indices = _select_diverse_match_indices(
            np.flatnonzero(preferred_keep),
            match_weights,
            xy1,
            preferred_quota,
            min_dist_px,
        )
    keep_fill = keep.copy()
    if len(preferred_indices):
        keep_fill[preferred_indices] = False
        fill_indices = _select_diverse_match_indices(
            np.flatnonzero(keep_fill),
            match_weights,
            xy1,
            max_valid_matches - len(preferred_indices),
            min_dist_px,
        ) if len(preferred_indices) < max_valid_matches else np.zeros(0, dtype=np.int32)
        valid_match_indices = np.sort(np.concatenate([preferred_indices, fill_indices]).astype(np.int32))
    else:
        valid_match_indices = _select_diverse_match_indices(
            np.flatnonzero(keep),
            match_weights,
            xy1,
            max_valid_matches,
            min_dist_px,
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
        "num_valid_3d_candidates": int(valid_candidate_count),
        "num_preferred_matches": int(preferred_keep.sum()),
        "num_selected_preferred_matches": int(len(preferred_indices)),
        "valid_match_limit": int(max_valid_matches),
        "match_spatial_separation_px": float(min_dist_px),
        "num_valid_3d_matches": int(len(valid_match_indices)),
        "valid_3d_pairs": int(len(valid_match_indices)),
        "valid_pairs": int(len(valid_match_indices)),
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
    anchor_debug = {
        "anchor_topk": int(getattr(cfg, "anchor_topk", 8)),
        "anchor_weight_boost": float(getattr(cfg, "anchor_weight_boost", 2.5)),
        "anchor_active_count_last_iter": 0,
        "anchor_active_count_max": 0,
        "anchor_rmse_last_m": math.inf,
        "anchor_rmse_best_m": math.inf,
        "anchor_refine_applied": False,
    }
    if T is not None and int(inliers_local.sum()) >= max(int(cfg.min_inliers), 4):
        T_refined, inliers_refined, anchor_debug = _anchor_local_refine(src3, tgt3, valid_weights, T, inliers_local, cfg)
        if T_refined is not None:
            T = T_refined
            inliers_local = inliers_refined
    timings["ransac_3d"] = time.perf_counter() - t_stage

    inlier_match_indices = valid_match_indices[inliers_local].astype(np.int32)
    debug.update(ransac_debug)
    debug.update(anchor_debug)
    debug["fit_method"] = "rigid_ransac"
    debug["small_sample_fallback_used"] = bool(fallback_used)
    debug["num_inliers"] = int(len(inlier_match_indices))
    debug["inlier_count"] = int(len(inlier_match_indices))
    debug["inlier_ratio"] = float(len(inlier_match_indices) / max(len(valid_match_indices), 1))
    debug["prior_rotation"] = T[:3, :3].tolist() if T is not None else None
    debug["prior_translation"] = T[:3, 3].tolist() if T is not None else None
    debug["rotation_deg"] = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)))) if T is not None else math.inf
    debug["translation_norm_m"] = float(np.linalg.norm(T[:3, 3])) if T is not None else math.inf
    debug["num_near_mask_inliers"] = int((keep_near_mask[inlier_match_indices]).sum()) if len(inlier_match_indices) else 0
    debug["near_mask_inlier_ratio"] = float(debug["num_near_mask_inliers"] / max(len(inlier_match_indices), 1))
    rmse = float(debug.get("ransac_rmse_m", math.inf))
    debug["rmse_m"] = float(rmse)
    req_min_inliers = int(cfg.small_sample_min_inliers) if fallback_used else int(cfg.min_inliers)
    req_min_ratio = float(cfg.small_sample_min_inlier_ratio) if fallback_used else float(cfg.min_inlier_ratio)
    req_max_rmse = float(cfg.small_sample_max_ransac_rmse_m) if fallback_used else float(cfg.max_ransac_rmse_m)
    near_mask_ok = debug["near_mask_inlier_ratio"] >= float(cfg.min_near_mask_inlier_ratio)
    prior_accept = False
    prior_strong = False
    if T is not None:
        t_stage = time.perf_counter()
        ref_surface_points, ref_surface_debug = _surface_points_from_mask_depth(
            before_depth_m, before_mask, intr, cfg.max_depth_m, int(cfg.dense_eval_max_points), int(cfg.random_seed) + 151
        )
        cur_surface_points, cur_surface_debug = _surface_points_from_mask_depth(
            after_depth_m, after_mask, intr, cfg.max_depth_m, int(cfg.dense_eval_max_points), int(cfg.random_seed) + 173
        )
        timings["dense_surface_points"] = time.perf_counter() - t_stage
        _emit_timing_log(
            cfg,
            "TAPIR dense surface extraction done "
            f"in {timings['dense_surface_points'] * 1000.0:.1f} ms "
            f"ref={int(ref_surface_debug.get('used_points', 0))} cur={int(cur_surface_debug.get('used_points', 0))}",
        )

        t_stage = time.perf_counter()
        prior_dense_rmse, prior_dense_fitness = _dense_transform_rmse(
            ref_surface_points,
            cur_surface_points,
            T,
            inlier_thresh_m=max(float(cfg.ransac_thresh_m), 0.015),
        )
        timings["dense_eval_rmse"] = time.perf_counter() - t_stage
        ref_diag = float(np.linalg.norm(np.ptp(ref_surface_points, axis=0))) if len(ref_surface_points) else 0.0
        cur_diag = float(np.linalg.norm(np.ptp(cur_surface_points, axis=0))) if len(cur_surface_points) else 0.0
        dense_scale = max(ref_diag, cur_diag, 1e-6)
        rel_dense_rmse = float(prior_dense_rmse / dense_scale) if np.isfinite(prior_dense_rmse) else math.inf

        t_stage = time.perf_counter()
        proj_metrics = _projected_surface_metrics(
            ref_surface_points,
            T,
            after_mask,
            intr,
            cfg.max_depth_m,
            int(cfg.projection_eval_downsample),
        )
        timings["projection_eval"] = time.perf_counter() - t_stage
        debug.update({
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
        })
        debug.update(proj_metrics)
        _emit_timing_log(
            cfg,
            "TAPIR dense eval done "
            f"rmse_ms={timings['dense_eval_rmse'] * 1000.0:.1f} "
            f"proj_ms={timings['projection_eval'] * 1000.0:.1f} "
            f"rel_rmse={debug.get('prior_dense_relative_rmse_eval', math.inf):.4f} "
            f"proj_iou={debug.get('prior_proj_iou', 0.0):.3f}",
        )
        prior_confidence, confidence_debug = _prior_confidence_from_debug(debug, cfg)
        borderline_accept, borderline_debug = _prior_accept_override(debug, cfg)
        prior_accept = bool(
            prior_confidence >= float(cfg.prior_confidence_accept_thresh)
            or borderline_accept
        )
        prior_strong = bool(prior_confidence >= float(cfg.prior_confidence_strong_thresh))
        debug.update(confidence_debug)
        debug.update(borderline_debug)
        debug["prior_confidence"] = float(prior_confidence)
        debug["prior_confidence_accept_thresh"] = float(cfg.prior_confidence_accept_thresh)
        debug["prior_confidence_strong_thresh"] = float(cfg.prior_confidence_strong_thresh)
        debug["prior_gate_proj_ok"] = bool(float(confidence_debug["prior_confidence_proj_score"]) >= 0.5)
        debug["prior_gate_rel_dense_ok"] = bool(float(confidence_debug["prior_confidence_dense_score"]) >= 0.5)
        debug["prior_gate_strong_3d_ok"] = bool(prior_strong)
        debug["prior_gate_decision"] = bool(prior_accept)
        debug["fallback_used"] = bool(not prior_accept)
        if bool(prior_accept):
            t_stage = time.perf_counter()
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
            timings["projection_support"] = time.perf_counter() - t_stage
            debug.update(projection_debug)
            _emit_timing_log(
                cfg,
                "TAPIR projection support done "
                f"in {timings['projection_support'] * 1000.0:.1f} ms "
                f"near_mask_ratio={float(debug.get('object_projection_near_mask_ratio', 0.0)):.3f}",
            )
        else:
            timings["projection_support"] = 0.0
            debug["object_projection_support_skipped"] = True
            debug["object_projection_near_mask_ratio"] = 0.0
    else:
        debug["prior_gate_proj_ok"] = False
        debug["prior_gate_rel_dense_ok"] = False
        debug["prior_gate_strong_3d_ok"] = False
        debug["prior_gate_decision"] = False
        debug["prior_gate_borderline_override"] = False
        debug["prior_confidence"] = 0.0
        debug["prior_confidence_accept_thresh"] = float(cfg.prior_confidence_accept_thresh)
        debug["prior_confidence_strong_thresh"] = float(cfg.prior_confidence_strong_thresh)
        debug["fallback_used"] = True
    debug["success"] = bool(
        T is not None
        and len(inlier_match_indices) >= req_min_inliers
        and debug["inlier_ratio"] >= req_min_ratio
        and rmse <= req_max_rmse
        and near_mask_ok
        and bool(prior_accept)
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
        elif not prior_accept:
            debug.setdefault("reason", "low prior confidence")
        else:
            debug.setdefault("reason", "tapir transform rejected")
        T = None
    else:
        debug["used_inside_icp"] = True
        debug["local_refine_only_used"] = True
        debug["reseed_skipped"] = True

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


def _estimate_tapir_init_transform_once(
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
    debug: dict[str, Any] = {"enabled": True, "method": "tapir", "backend": str(getattr(cfg, "backend", "") or "")}
    if str(getattr(cfg, "backend", "") or "").lower() in {"remote_tapnet", "remote", "tapir"}:
        debug.update(_remote_debug_base(cfg, 0))
    timings: dict[str, float] = {}
    t_total = time.perf_counter()
    _emit_timing_log(cfg, "TAPIR init begin")
    try:
        t_stage = time.perf_counter()
        query_xy, sample_debug = _sample_query_points(before_rgb, before_depth_m, before_mask, cfg)
        timings["sample_queries"] = time.perf_counter() - t_stage
        debug.update(sample_debug)
        debug["target_crop_bbox_xyxy"] = _mask_bbox_xyxy(after_mask)
        _emit_timing_log(
            cfg,
            f"TAPIR sample_queries done in {timings['sample_queries'] * 1000.0:.1f} ms, selected={len(query_xy)}",
        )

        t_stage = time.perf_counter()
        kpts0, kpts1, track_quality, track_debug = _track_points(before_rgb, after_rgb, query_xy, cfg)
        timings["track"] = time.perf_counter() - t_stage
        debug.update(track_debug)
        track_remote_roundtrip_s = float(track_debug.get("tapir_rpc_total_ms", 0.0)) / 1000.0
        timings["track_remote_roundtrip"] = min(float(timings["track"]), max(0.0, track_remote_roundtrip_s))
        timings["track_no_remote_roundtrip"] = max(0.0, float(timings["track"]) - float(timings["track_remote_roundtrip"]))
        track_no_rpc_io_s = float(track_debug.get("tapir_rpc_server_compute_ms", 0.0)) / 1000.0
        if track_no_rpc_io_s <= 0.0:
            track_no_rpc_io_s = float(timings["track"])
        timings["track_no_rpc_io"] = track_no_rpc_io_s
        timings["track_rpc_io_overhead"] = max(0.0, float(timings["track"]) - track_no_rpc_io_s)
        if "tapir" in debug and isinstance(debug["tapir"], dict):
            debug["tapir"]["latency_ms"] = int(round(timings["track"] * 1000.0))
            debug["tapir"]["track_no_rpc_io_ms"] = int(round(track_no_rpc_io_s * 1000.0))
            debug["tapir"]["track_no_remote_roundtrip_ms"] = int(round(float(timings["track_no_remote_roundtrip"]) * 1000.0))
        debug["tapir_latency_ms"] = int(round(timings["track"] * 1000.0))
        debug["tapir_latency_no_rpc_io_ms"] = int(round(track_no_rpc_io_s * 1000.0))
        debug["tapir_latency_no_remote_roundtrip_ms"] = int(round(float(timings["track_no_remote_roundtrip"]) * 1000.0))
        _emit_timing_log(
            cfg,
            f"TAPIR tracking done in {timings['track'] * 1000.0:.1f} ms, kept={len(kpts0)}",
        )

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
        timings["total_no_remote_roundtrip"] = max(
            0.0,
            float(timings["total"]) - float(timings.get("track_remote_roundtrip", 0.0)),
        )
        timings["total_no_rpc_io"] = max(
            0.0,
            float(timings["total"]) - float(timings.get("track", 0.0)) + float(timings.get("track_no_rpc_io", 0.0)),
        )
        result.debug["timings_s"] = timings
        result.debug["tapir_init_latency_ms"] = int(round(float(timings["total"]) * 1000.0))
        result.debug["tapir_init_latency_no_remote_roundtrip_ms"] = int(round(float(timings["total_no_remote_roundtrip"]) * 1000.0))
        result.debug["tapir_init_latency_no_rpc_io_ms"] = int(round(float(timings["total_no_rpc_io"]) * 1000.0))
        _emit_timing_log(
            cfg,
            "TAPIR init done "
            + " ".join(f"{key}={float(value) * 1000.0:.1f}ms" for key, value in timings.items()),
        )
        return result
    except Exception as exc:
        timings["total"] = time.perf_counter() - t_total
        timings["total_no_remote_roundtrip"] = max(
            0.0,
            float(timings["total"]) - float(timings.get("track_remote_roundtrip", 0.0)),
        )
        timings["total_no_rpc_io"] = max(
            0.0,
            float(timings["total"]) - float(timings.get("track", 0.0)) + float(timings.get("track_no_rpc_io", 0.0)),
        )
        exc_debug = getattr(exc, "debug", None)
        if isinstance(exc_debug, dict):
            debug.update(exc_debug)
        debug["timings_s"] = timings
        debug["tapir_init_latency_ms"] = int(round(float(timings["total"]) * 1000.0))
        debug["tapir_init_latency_no_remote_roundtrip_ms"] = int(round(float(timings["total_no_remote_roundtrip"]) * 1000.0))
        debug["tapir_init_latency_no_rpc_io_ms"] = int(round(float(timings["total_no_rpc_io"]) * 1000.0))
        debug.update({"success": False, "error": str(exc), "stage": "tapir_track"})
        _emit_timing_log(
            cfg,
            f"TAPIR init failed after {timings['total'] * 1000.0:.1f} ms: {exc}",
        )
        return TapirInitResult(success=False, debug=debug)


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
    t_total = time.perf_counter()
    attempts: list[TapirInitResult] = []
    primary_result = _estimate_tapir_init_transform_once(
        before_rgb=before_rgb,
        before_depth_m=before_depth_m,
        before_mask=before_mask,
        after_rgb=after_rgb,
        after_depth_m=after_depth_m,
        after_mask=after_mask,
        intr=intr,
        cfg=cfg,
    )
    attempts.append(primary_result)
    best_result = primary_result
    retry_needed, retry_reasons = _tapir_retry_needed(primary_result.debug or {}, cfg)
    tried_modes = {str((primary_result.debug or {}).get("sampling_mode", getattr(cfg, "sampling_mode", "")) or "")}
    if retry_needed:
        _emit_timing_log(cfg, f"TAPIR retry triggered reasons={retry_reasons}")
        for retry_mode in tuple(getattr(cfg, "sampling_retry_modes", ()) or ()):
            mode = str(retry_mode or "").strip().lower()
            if not mode or mode in tried_modes:
                continue
            tried_modes.add(mode)
            retry_cfg = replace(cfg, sampling_mode=mode)
            retry_result = _estimate_tapir_init_transform_once(
                before_rgb=before_rgb,
                before_depth_m=before_depth_m,
                before_mask=before_mask,
                after_rgb=after_rgb,
                after_depth_m=after_depth_m,
                after_mask=after_mask,
                intr=intr,
                cfg=retry_cfg,
            )
            attempts.append(retry_result)
            if _tapir_result_rank(retry_result) > _tapir_result_rank(best_result):
                best_result = retry_result

    aggregate_timings: dict[str, float] = {}
    attempt_summaries: list[dict[str, Any]] = []
    best_id = id(best_result)
    for attempt in attempts:
        attempt_debug = dict(attempt.debug or {})
        attempt_timings = dict(attempt_debug.get("timings_s") or {})
        for key, value in attempt_timings.items():
            if key in {"total", "total_no_remote_roundtrip", "total_no_rpc_io"}:
                continue
            aggregate_timings[key] = float(aggregate_timings.get(key, 0.0)) + float(value)
        attempt_summaries.append(
            {
                "sampling_mode": str(attempt_debug.get("sampling_mode", "")),
                "success": bool(attempt_debug.get("success", attempt.success)),
                "num_inliers": int(attempt_debug.get("num_inliers", attempt_debug.get("inlier_count", 0)) or 0),
                "prior_proj_iou": float(attempt_debug.get("prior_proj_iou", 0.0) or 0.0),
                "prior_proj_precision": float(attempt_debug.get("prior_proj_precision", 0.0) or 0.0),
                "prior_dense_relative_rmse_eval": float(attempt_debug.get("prior_dense_relative_rmse_eval", math.inf) or math.inf),
                "ransac_rmse_m": float(attempt_debug.get("ransac_rmse_m", attempt_debug.get("rmse_m", math.inf)) or math.inf),
                "selected": bool(id(attempt) == best_id),
            }
        )

    total_elapsed = time.perf_counter() - t_total
    total_remote_roundtrip = float(aggregate_timings.get("track_remote_roundtrip", 0.0))
    total_track = float(aggregate_timings.get("track", 0.0))
    total_track_no_rpc_io = float(aggregate_timings.get("track_no_rpc_io", 0.0))
    aggregate_timings["total"] = float(total_elapsed)
    aggregate_timings["total_no_remote_roundtrip"] = max(0.0, float(total_elapsed) - total_remote_roundtrip)
    aggregate_timings["total_no_rpc_io"] = max(0.0, float(total_elapsed) - total_track + total_track_no_rpc_io)

    final_debug = dict(best_result.debug or {})
    final_debug["sampling_attempts"] = attempt_summaries
    final_debug["sampling_retry_triggered"] = bool(len(attempts) > 1)
    final_debug["sampling_retry_reasons"] = list(retry_reasons)
    final_debug["timings_s"] = aggregate_timings
    final_debug["tapir_init_latency_ms"] = int(round(float(aggregate_timings["total"]) * 1000.0))
    final_debug["tapir_init_latency_no_remote_roundtrip_ms"] = int(round(float(aggregate_timings["total_no_remote_roundtrip"]) * 1000.0))
    final_debug["tapir_init_latency_no_rpc_io_ms"] = int(round(float(aggregate_timings["total_no_rpc_io"]) * 1000.0))
    best_result.debug = final_debug
    return best_result


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
