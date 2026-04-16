from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


def mask_contains(mask: np.ndarray, xy: np.ndarray, margin_px: int) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = (m > 0).astype(np.uint8)
    if margin_px != 0:
        k = 2 * abs(int(margin_px)) + 1
        kernel = np.ones((k, k), dtype=np.uint8)
        if margin_px > 0:
            m = cv2.dilate(m, kernel, iterations=1)
        else:
            m = cv2.erode(m, kernel, iterations=1)
    h, w = m.shape[:2]
    u = np.round(xy[:, 0]).astype(np.int32)
    v = np.round(xy[:, 1]).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    keep = np.zeros(len(xy), dtype=bool)
    keep[inside] = m[v[inside], u[inside]] > 0
    return keep


def roi_contains(mask: np.ndarray, xy: np.ndarray, bbox_margin_px: int) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = m > 0
    h, w = m.shape[:2]
    keep = np.zeros(len(xy), dtype=bool)
    if not m.any():
        return keep
    ys, xs = np.nonzero(m)
    margin = max(int(bbox_margin_px), 0)
    x1 = max(int(xs.min()) - margin, 0)
    x2 = min(int(xs.max()) + margin, w - 1)
    y1 = max(int(ys.min()) - margin, 0)
    y2 = min(int(ys.max()) + margin, h - 1)
    u = np.round(xy[:, 0]).astype(np.int32)
    v = np.round(xy[:, 1]).astype(np.int32)
    keep = (u >= x1) & (u <= x2) & (v >= y1) & (v <= y2)
    return keep


def mask_distance_values(mask: np.ndarray, xy: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = m > 0
    h, w = m.shape[:2]
    outside = (~m).astype(np.uint8)
    dist = cv2.distanceTransform(outside, cv2.DIST_L2, 3).astype(np.float32)
    u = np.round(xy[:, 0]).astype(np.int32)
    v = np.round(xy[:, 1]).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    vals = np.full(len(xy), np.inf, dtype=np.float32)
    vals[inside] = dist[v[inside], u[inside]]
    return vals


def sample_depth(depth_m: np.ndarray, xy: np.ndarray, max_depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape[:2]
    u = np.round(xy[:, 0]).astype(np.int32)
    v = np.round(xy[:, 1]).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    z = np.zeros(len(xy), dtype=np.float64)
    z[inside] = depth_m[v[inside], u[inside]].astype(np.float64)
    valid = inside & np.isfinite(z) & (z > 1e-5) & (z < float(max_depth_m))
    return z, valid


def backproject(xy: np.ndarray, z: np.ndarray, intr: Any) -> np.ndarray:
    u = xy[:, 0].astype(np.float64)
    v = xy[:, 1].astype(np.float64)
    z = z.astype(np.float64)
    x = (u - float(intr.cx)) * z / float(intr.fx)
    y = (v - float(intr.cy)) * z / float(intr.fy)
    return np.stack([x, y, z], axis=1)


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
    m = np.asarray(before_mask)
    if m.ndim == 3:
        m = m[:, :, 0]
    m = m > 0
    depth = np.asarray(before_depth_m)
    valid_depth = np.isfinite(depth) & (depth > 1e-5) & (depth < float(max_depth_m))
    ys, xs = np.nonzero(m & valid_depth)
    total = int(len(xs))
    if total == 0:
        return {
            "object_projection_sampled": 0,
            "object_projection_valid": 0,
            "object_projection_mask_ratio": 0.0,
            "object_projection_near_mask_ratio": 0.0,
            "object_projection_reason": "no before object depth samples",
        }

    max_samples = max(int(max_samples), 1)
    if total > max_samples:
        rng = np.random.default_rng(int(random_seed))
        ids = rng.choice(total, max_samples, replace=False)
        xs = xs[ids]
        ys = ys[ids]

    xy = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    z = depth[ys, xs].astype(np.float64)
    pts = backproject(xy, z, intr)
    T = np.asarray(T_before_to_after, dtype=np.float64).reshape(4, 4)
    pts_t = (T[:3, :3] @ pts.T).T + T[:3, 3]
    zt = pts_t[:, 2]
    front = np.isfinite(pts_t).all(axis=1) & (zt > 1e-5) & (zt < float(max_depth_m))
    if not front.any():
        return {
            "object_projection_sampled": int(len(xy)),
            "object_projection_valid": 0,
            "object_projection_mask_ratio": 0.0,
            "object_projection_near_mask_ratio": 0.0,
            "object_projection_reason": "no projected object samples in front of camera",
        }

    xy_proj = np.empty((int(front.sum()), 2), dtype=np.float32)
    pts_front = pts_t[front]
    xy_proj[:, 0] = (pts_front[:, 0] * float(intr.fx) / pts_front[:, 2]) + float(intr.cx)
    xy_proj[:, 1] = (pts_front[:, 1] * float(intr.fy) / pts_front[:, 2]) + float(intr.cy)
    mask_hits = mask_contains(after_mask, xy_proj, 0)
    near_hits = mask_distance_values(after_mask, xy_proj) <= float(near_mask_px)
    return {
        "object_projection_sampled": int(len(xy)),
        "object_projection_valid": int(len(xy_proj)),
        "object_projection_mask_hits": int(mask_hits.sum()),
        "object_projection_near_mask_hits": int(near_hits.sum()),
        "object_projection_mask_ratio": float(mask_hits.mean()) if len(mask_hits) else 0.0,
        "object_projection_near_mask_ratio": float(near_hits.mean()) if len(near_hits) else 0.0,
    }


def rigid_transform_svd(src: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    cs = src.mean(axis=0)
    ct = tgt.mean(axis=0)
    x = src - cs
    y = tgt - ct
    h = x.T @ y
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = ct - r @ cs
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t
    return T


def ransac_rigid_transform(
    src: np.ndarray,
    tgt: np.ndarray,
    cfg: Any,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, dict[str, Any]]:
    n = len(src)
    if n < max(3, cfg.min_inliers):
        return None, np.zeros(n, dtype=bool), {"reason": "not enough 3D correspondences"}
    rng = np.random.default_rng(int(cfg.random_seed))
    best_inliers = np.zeros(n, dtype=bool)
    best_rmse = math.inf
    best_T = None
    sample_size = 3
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != n or not np.isfinite(weights).all() or weights.sum() <= 1e-9:
            weights = None
        else:
            weights = np.maximum(weights, 1e-6)
            weights = weights / weights.sum()
    for _ in range(int(cfg.ransac_iters)):
        ids = rng.choice(n, sample_size, replace=False, p=weights)
        try:
            T = rigid_transform_svd(src[ids], tgt[ids])
        except np.linalg.LinAlgError:
            continue
        src_t = (T[:3, :3] @ src.T).T + T[:3, 3]
        err = np.linalg.norm(src_t - tgt, axis=1)
        inliers = err < float(cfg.ransac_thresh_m)
        count = int(inliers.sum())
        if count < int(cfg.min_inliers):
            continue
        rmse = float(np.sqrt(np.mean(err[inliers] ** 2)))
        weighted_count = float(weights[inliers].sum() * n) if weights is not None else float(count)
        best_weighted_count = float(weights[best_inliers].sum() * n) if weights is not None else float(best_inliers.sum())
        if weighted_count > best_weighted_count or (
            abs(weighted_count - best_weighted_count) < 1e-6 and rmse < best_rmse
        ):
            best_inliers = inliers
            best_rmse = rmse
            best_T = T
    if best_T is None:
        return None, best_inliers, {"reason": "ransac found no consensus"}
    best_T = rigid_transform_svd(src[best_inliers], tgt[best_inliers])
    src_t = (best_T[:3, :3] @ src.T).T + best_T[:3, 3]
    err = np.linalg.norm(src_t - tgt, axis=1)
    best_inliers = err < float(cfg.ransac_thresh_m)
    rmse = float(np.sqrt(np.mean(err[best_inliers] ** 2))) if best_inliers.any() else math.inf
    return best_T, best_inliers, {
        "ransac_rmse_m": rmse,
        "ransac_thresh_m": float(cfg.ransac_thresh_m),
    }


def select_valid_matches(
    keep: np.ndarray,
    weights: np.ndarray,
    max_valid_matches: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = np.nonzero(keep)[0].astype(np.int32)
    if len(idx) <= int(max_valid_matches):
        return idx
    w = np.asarray(weights[idx], dtype=np.float64)
    if not np.isfinite(w).all() or w.sum() <= 1e-9:
        order = rng.permutation(len(idx))[: int(max_valid_matches)]
    else:
        w = w / w.sum()
        order = rng.choice(len(idx), int(max_valid_matches), replace=False, p=w)
    return idx[order].astype(np.int32)
