from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def backproject(xy: np.ndarray, z: np.ndarray, intr: Any) -> np.ndarray:
    xy_arr = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    if len(xy_arr) != len(z_arr):
        raise ValueError("xy and z must have the same length")
    x = (xy_arr[:, 0] - float(intr.cx)) * z_arr / float(intr.fx)
    y = (xy_arr[:, 1] - float(intr.cy)) * z_arr / float(intr.fy)
    return np.column_stack([x, y, z_arr]).astype(np.float64)


def _ensure_bool_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr > 0


def _sample_mask(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    mask_bool = _ensure_bool_mask(mask)
    h, w = mask_bool.shape[:2]
    pts = np.rint(np.asarray(xy, dtype=np.float64)).astype(np.int64)
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    out = np.zeros(len(pts), dtype=bool)
    if np.any(inside):
        out[inside] = mask_bool[pts[inside, 1], pts[inside, 0]]
    return out


def mask_contains(mask: np.ndarray, xy: np.ndarray, margin_px: int = 0) -> np.ndarray:
    mask_u8 = (_ensure_bool_mask(mask).astype(np.uint8) * 255)
    margin = max(0, int(margin_px))
    if margin > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        mask_u8 = cv2.dilate(mask_u8, kernel, iterations=1)
    return _sample_mask(mask_u8, xy)


def roi_contains(mask: np.ndarray, xy: np.ndarray, margin_px: int = 0) -> np.ndarray:
    mask_bool = _ensure_bool_mask(mask)
    ys, xs = np.nonzero(mask_bool)
    pts = np.rint(np.asarray(xy, dtype=np.float64)).astype(np.int64)
    if len(xs) == 0 or len(pts) == 0:
        return np.zeros(len(pts), dtype=bool)
    margin = max(0, int(margin_px))
    x0 = int(xs.min()) - margin
    x1 = int(xs.max()) + margin
    y0 = int(ys.min()) - margin
    y1 = int(ys.max()) + margin
    return (
        (pts[:, 0] >= x0)
        & (pts[:, 0] <= x1)
        & (pts[:, 1] >= y0)
        & (pts[:, 1] <= y1)
    )


def mask_distance_values(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    mask_bool = _ensure_bool_mask(mask)
    inverse = (~mask_bool).astype(np.uint8)
    dist = cv2.distanceTransform(inverse, cv2.DIST_L2, 3)
    pts = np.rint(np.asarray(xy, dtype=np.float64)).astype(np.int64)
    h, w = dist.shape[:2]
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    out = np.full(len(pts), np.inf, dtype=np.float64)
    if np.any(inside):
        out[inside] = dist[pts[inside, 1], pts[inside, 0]].astype(np.float64)
    return out


def sample_depth(depth_m: np.ndarray, xy: np.ndarray, max_depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth_m, dtype=np.float32)
    pts = np.rint(np.asarray(xy, dtype=np.float64)).astype(np.int64)
    h, w = depth.shape[:2]
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    z = np.zeros(len(pts), dtype=np.float32)
    valid = np.zeros(len(pts), dtype=bool)
    if np.any(inside):
        sampled = depth[pts[inside, 1], pts[inside, 0]]
        ok = np.isfinite(sampled) & (sampled > 1e-6) & (sampled < float(max_depth_m))
        z[inside] = sampled
        valid[inside] = ok
    return z.astype(np.float64), valid


def select_valid_matches(keep: np.ndarray, weights: np.ndarray, max_count: int, rng: np.random.Generator) -> np.ndarray:
    keep_arr = np.asarray(keep, dtype=bool).reshape(-1)
    valid_idx = np.flatnonzero(keep_arr)
    if len(valid_idx) <= max_count:
        return valid_idx.astype(np.int32)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)[valid_idx]
    weight_arr = np.clip(weight_arr, 1e-8, None)
    probs = weight_arr / weight_arr.sum()
    selected = rng.choice(valid_idx, size=int(max_count), replace=False, p=probs)
    return np.sort(selected.astype(np.int32))


def _estimate_rigid_transform(src: np.ndarray, tgt: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray | None:
    src_pts = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    tgt_pts = np.asarray(tgt, dtype=np.float64).reshape(-1, 3)
    if len(src_pts) != len(tgt_pts) or len(src_pts) < 3:
        return None
    if weights is None:
        w = np.ones(len(src_pts), dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(w) != len(src_pts):
            return None
        w = np.clip(w, 1e-8, None)
    w = w / w.sum()
    src_center = np.sum(src_pts * w[:, None], axis=0)
    tgt_center = np.sum(tgt_pts * w[:, None], axis=0)
    src_zero = src_pts - src_center
    tgt_zero = tgt_pts - tgt_center
    h_mat = (src_zero * w[:, None]).T @ tgt_zero
    u_mat, _, vt_mat = np.linalg.svd(h_mat)
    rot = vt_mat.T @ u_mat.T
    if np.linalg.det(rot) < 0.0:
        vt_mat[-1, :] *= -1.0
        rot = vt_mat.T @ u_mat.T
    trans = tgt_center - rot @ src_center
    tform = np.eye(4, dtype=np.float64)
    tform[:3, :3] = rot
    tform[:3, 3] = trans
    return tform


def estimate_rigid_transform(src: np.ndarray, tgt: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray | None:
    return _estimate_rigid_transform(src, tgt, weights)


def filter_projected_uv(uv: np.ndarray, image_shape: tuple[int, int], connect_radius_px: int = 3) -> np.ndarray:
    pts = np.asarray(uv, dtype=np.int32).reshape(-1, 2)
    if len(pts) <= 4:
        return pts
    h, w = image_shape
    inside = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[inside]
    if len(pts) <= 4:
        return pts
    pts = np.unique(pts, axis=0)
    if len(pts) <= 4:
        return pts

    sparse = np.zeros((h, w), dtype=np.uint8)
    sparse[pts[:, 1], pts[:, 0]] = 255
    radius = max(1, int(connect_radius_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    connected = cv2.dilate(sparse, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(connected, connectivity=8)
    if num_labels <= 2:
        return pts

    point_labels = labels[pts[:, 1], pts[:, 0]]
    valid_labels = point_labels[point_labels > 0]
    if len(valid_labels) == 0:
        return pts
    counts = np.bincount(valid_labels, minlength=num_labels)
    best_label = int(np.argmax(counts[1:]) + 1)
    keep = point_labels == best_label
    keep_count = int(keep.sum())
    if keep_count < max(8, int(0.5 * len(pts))):
        return pts
    return pts[keep]


def projected_uv_to_mask(
    uv: np.ndarray,
    image_shape: tuple[int, int],
    point_radius_px: int = 1,
    connect_radius_px: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    filtered = filter_projected_uv(uv, image_shape, connect_radius_px=connect_radius_px)
    mask = np.zeros((h, w), dtype=np.uint8)
    if len(filtered) == 0:
        return mask, filtered
    mask[filtered[:, 1], filtered[:, 0]] = 255
    point_radius = max(0, int(point_radius_px))
    if point_radius > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (point_radius * 2 + 1, point_radius * 2 + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)
    close_radius = max(1, int(connect_radius_px))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_radius * 2 + 1, close_radius * 2 + 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep_label = int(np.argmax(areas) + 1)
        mask = np.where(labels == keep_label, 255, 0).astype(np.uint8)
    return mask, filtered


def ransac_rigid_transform(src: np.ndarray, tgt: np.ndarray, cfg: Any, weights: np.ndarray | None = None) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    src_pts = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    tgt_pts = np.asarray(tgt, dtype=np.float64).reshape(-1, 3)
    n = len(src_pts)
    empty_inliers = np.zeros(n, dtype=bool)
    if n < 3 or n != len(tgt_pts):
        return None, empty_inliers, {
            "ransac_success": False,
            "ransac_num_hypotheses": 0,
            "ransac_num_inliers": 0,
            "ransac_inlier_ratio": 0.0,
            "ransac_rmse_m": math.inf,
        }

    rng = np.random.default_rng(int(getattr(cfg, "random_seed", 0)) + 23)
    thresh = float(getattr(cfg, "ransac_thresh_m", 0.01))
    iters = max(32, int(getattr(cfg, "ransac_iters", 800)))
    min_iters = min(iters, 32)
    confidence = float(getattr(cfg, "ransac_confidence", 0.995) or 0.995)
    confidence = min(max(confidence, 1e-6), 1.0 - 1e-6)
    sample_size = min(4, n)

    if weights is None:
        sample_probs = np.full(n, 1.0 / n, dtype=np.float64)
        fit_weights = np.ones(n, dtype=np.float64)
    else:
        fit_weights = np.clip(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-8, None)
        sample_probs = fit_weights / fit_weights.sum()

    best_tform: np.ndarray | None = None
    best_inliers = empty_inliers.copy()
    best_score = (-1, -math.inf)
    effective_iters = iters
    actual_iters = 0

    for _ in range(iters):
        actual_iters += 1
        sample_idx = rng.choice(n, size=sample_size, replace=False, p=sample_probs)
        tform = _estimate_rigid_transform(src_pts[sample_idx], tgt_pts[sample_idx])
        if tform is None:
            continue
        src_t = (src_pts @ tform[:3, :3].T) + tform[:3, 3]
        err = np.linalg.norm(src_t - tgt_pts, axis=1)
        inliers = err <= thresh
        count = int(inliers.sum())
        if count < 3:
            continue
        score = (count, float(-err[inliers].mean()))
        if score > best_score:
            best_score = score
            best_tform = tform
            best_inliers = inliers
            if count >= n:
                effective_iters = actual_iters
                break
            inlier_ratio = float(count) / max(float(n), 1.0)
            success_prob = inlier_ratio ** sample_size
            if 0.0 < success_prob < 1.0:
                required = math.log(1.0 - confidence) / math.log(1.0 - success_prob)
                effective_iters = min(effective_iters, max(min_iters, int(math.ceil(required))))
        if best_tform is not None and actual_iters >= effective_iters:
            break

    if best_tform is None or int(best_inliers.sum()) < 3:
        return None, empty_inliers, {
            "ransac_success": False,
            "ransac_num_hypotheses": actual_iters,
            "ransac_num_inliers": 0,
            "ransac_inlier_ratio": 0.0,
            "ransac_rmse_m": math.inf,
        }

    refined = _estimate_rigid_transform(
        src_pts[best_inliers],
        tgt_pts[best_inliers],
        None if weights is None else fit_weights[best_inliers],
    )
    if refined is not None:
        best_tform = refined
        src_t = (src_pts @ best_tform[:3, :3].T) + best_tform[:3, 3]
        err = np.linalg.norm(src_t - tgt_pts, axis=1)
        best_inliers = err <= thresh
    else:
        src_t = (src_pts @ best_tform[:3, :3].T) + best_tform[:3, 3]
        err = np.linalg.norm(src_t - tgt_pts, axis=1)

    inlier_err = err[best_inliers]
    rmse = float(np.sqrt(np.mean(inlier_err * inlier_err))) if len(inlier_err) else math.inf
    return best_tform, best_inliers, {
        "ransac_success": True,
        "ransac_num_hypotheses": actual_iters,
        "ransac_num_inliers": int(best_inliers.sum()),
        "ransac_inlier_ratio": float(best_inliers.mean()) if len(best_inliers) else 0.0,
        "ransac_rmse_m": rmse,
    }


def object_projection_support(
    before_depth_m: np.ndarray,
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    T_before_to_after: np.ndarray,
    intr: Any,
    max_depth_m: float,
    near_mask_px: float,
    max_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    mask_bool = _ensure_bool_mask(before_mask)
    ys, xs = np.nonzero(mask_bool)
    total = len(xs)
    if total == 0:
        return {
            "object_projection_pixels": 0,
            "object_projection_precision": 0.0,
            "object_projection_iou": 0.0,
            "object_projection_near_mask_ratio": 0.0,
        }

    if total > int(max_samples) > 0:
        rng = np.random.default_rng(int(random_seed))
        keep = rng.choice(total, int(max_samples), replace=False)
        xs = xs[keep]
        ys = ys[keep]
    z = before_depth_m[ys, xs]
    valid = np.isfinite(z) & (z > 1e-6) & (z < float(max_depth_m))
    if not np.any(valid):
        return {
            "object_projection_pixels": 0,
            "object_projection_precision": 0.0,
            "object_projection_iou": 0.0,
            "object_projection_near_mask_ratio": 0.0,
        }

    xy = np.column_stack([xs[valid], ys[valid]]).astype(np.float64)
    src_pts = backproject(xy, z[valid], intr)
    tform = np.asarray(T_before_to_after, dtype=np.float64).reshape(4, 4)
    tgt_pts = (src_pts @ tform[:3, :3].T) + tform[:3, 3]
    z_t = tgt_pts[:, 2]
    front = np.isfinite(tgt_pts).all(axis=1) & (z_t > 1e-6) & (z_t < float(max_depth_m))
    if not np.any(front):
        return {
            "object_projection_pixels": 0,
            "object_projection_precision": 0.0,
            "object_projection_iou": 0.0,
            "object_projection_near_mask_ratio": 0.0,
        }
    pts = tgt_pts[front]
    u = np.round((pts[:, 0] * float(intr.fx) / pts[:, 2]) + float(intr.cx)).astype(np.int32)
    v = np.round((pts[:, 1] * float(intr.fy) / pts[:, 2]) + float(intr.cy)).astype(np.int32)
    mask_after = _ensure_bool_mask(after_mask)
    h, w = mask_after.shape[:2]
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    if not np.any(inside):
        return {
            "object_projection_pixels": 0,
            "object_projection_precision": 0.0,
            "object_projection_iou": 0.0,
            "object_projection_near_mask_ratio": 0.0,
        }

    uv = np.column_stack([u[inside], v[inside]]).astype(np.int32)
    proj_mask, uv_filtered = projected_uv_to_mask(uv, (h, w), point_radius_px=1, connect_radius_px=3)
    if len(uv_filtered) == 0:
        return {
            "object_projection_pixels": 0,
            "object_projection_precision": 0.0,
            "object_projection_iou": 0.0,
            "object_projection_near_mask_ratio": 0.0,
        }

    pred = proj_mask > 0
    gt = mask_after
    inter = int(np.logical_and(pred, gt).sum())
    pred_area = int(pred.sum())
    union = int(np.logical_or(pred, gt).sum())
    precision = float(inter / max(pred_area, 1))
    iou = float(inter / max(union, 1))

    dist_to_mask = cv2.distanceTransform((~gt).astype(np.uint8), cv2.DIST_L2, 3)
    near_vals = dist_to_mask[uv_filtered[:, 1], uv_filtered[:, 0]].astype(np.float64)
    near_ratio = float(np.mean(near_vals <= float(near_mask_px))) if len(near_vals) else 0.0
    return {
        "object_projection_pixels": pred_area,
        "object_projection_precision": precision,
        "object_projection_iou": iou,
        "object_projection_near_mask_ratio": near_ratio,
    }
