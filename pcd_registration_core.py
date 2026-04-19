from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]


# ============================================================
# Partial RGB-D object registration (partial-to-partial)
#
# Scenario A:
# - head camera sees a partial RGB-D object observation
# - wrist camera sees another partial RGB-D object observation
# - both RGB-D inputs are already object-only (segmented)
# - output is the rigid transform aligning head observation to wrist observation
#
# Main pipeline:
# 1) mask cleanup / depth-edge filtering
# 2) RGB-D -> colored point cloud
# 3) point cloud cleanup + normals
# 4) move both clouds into base frame
# 5) PCA-based weak initialization
# 6) local multi-hypothesis search around init
# 7) visibility-aware scoring in wrist camera view
# 8) GICP refinement
# 9) optional colored ICP refinement
#
# Requirements:
#   pip install open3d opencv-python numpy
# ============================================================


@dataclass
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class RegistrationConfig:
    depth_scale: float = 1000.0       # e.g. depth in millimeters -> meters
    depth_min_m: float = 0.00
    depth_max_m: float = 2.0

    # mask / depth cleanup
    erode_kernel: int = 5
    erode_iters: int = 1
    depth_edge_thresh_m: float = 0.01
    remove_depth_edge_points: bool = True

    # point cloud cleanup
    voxel_size: float = 0.003         # 3 mm
    outlier_nb_neighbors: int = 30
    outlier_std_ratio: float = 1.0
    radius_outlier_radius: float = 0.008
    radius_outlier_min_neighbors: int = 6
    iqr_multiplier: float = 1.5
    normal_radius: float = 0.01
    normal_max_nn: int = 30

    # initialization / hypothesis search
    use_pca_init: bool = True
    use_fpfh_ransac_init: bool = True
    fpfh_ransac_voxel_size: float = 0.006
    fpfh_normal_radius_factor: float = 2.5
    fpfh_feature_radius_factor: float = 5.0
    fpfh_ransac_max_corr_factor: float = 2.5
    fpfh_ransac_max_iter: int = 60000
    fpfh_ransac_confidence: float = 0.999
    fpfh_ransac_min_fitness: float = 0.05
    search_rx_deg: Tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)
    search_ry_deg: Tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)
    search_rz_deg: Tuple[float, ...] = (-30.0, -15.0, 0.0, 15.0, 30.0)
    search_tx_m: Tuple[float, ...] = (-0.01, 0.0, 0.01)
    search_ty_m: Tuple[float, ...] = (-0.01, 0.0, 0.01)
    search_tz_m: Tuple[float, ...] = (-0.01, 0.0, 0.01)

    # visibility-aware scoring
    depth_consistency_thresh_m: float = 0.008
    color_consistency_weight: float = 0.15
    normal_consistency_weight: float = 0.15
    depth_consistency_weight: float = 0.70
    min_visible_points: int = 80

    # adaptive elongated-object prior
    axis_prior_enable: bool = True
    axis_prior_min_points: int = 120
    axis_prior_min_ratio: float = 3.0
    axis_prior_full_ratio: float = 7.0
    axis_prior_max_weight: float = 0.28
    axis_prior_center_tol_m: float = 0.012
    axis_prior_length_tol_ratio: float = 0.25

    # ICP refinement
    gicp_max_corr_dist: float = 0.04
    gicp_max_iter: int = 60
    use_colored_icp: bool = False
    colored_icp_max_corr_dist: float = 0.03
    colored_icp_max_iter: int = 30


def _require_open3d() -> None:
    if o3d is None:
        raise ImportError("open3d is required for point-cloud registration. Please install open3d first.")


# -------------------------
# Basic transform utilities
# -------------------------
def to_homogeneous(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    assert T.shape == (4, 4)
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts_h = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    pts_t = (T @ pts_h.T).T
    return pts_t[:, :3]


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def rotx(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def roty(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotz(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def euler_xyz_deg_to_R(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)
    return rotz(rz) @ roty(ry) @ rotx(rx)


# -------------------------
# Image / mask processing
# -------------------------
def ensure_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255
    return mask


def erode_mask(mask: np.ndarray, kernel_size: int = 5, iterations: int = 1) -> np.ndarray:
    mask = ensure_uint8_mask(mask)
    if kernel_size <= 1 or iterations <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=iterations)


def compute_depth_edge_mask(depth_m: np.ndarray, valid_mask: np.ndarray, thresh_m: float) -> np.ndarray:
    """
    Detect strong local depth discontinuities. Returns boolean mask where True means edge-like.
    """
    depth = depth_m.copy()
    depth[~valid_mask] = 0.0
    gx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    g = np.sqrt(gx * gx + gy * gy)
    return g > thresh_m


# -------------------------
# RGB-D -> point cloud
# -------------------------
def rgbd_to_pointcloud(
    rgb: np.ndarray,
    depth: np.ndarray,
    mask: np.ndarray,
    intr: CameraIntrinsics,
    cfg: RegistrationConfig,
) -> o3d.geometry.PointCloud:
    """
    Create object-only point cloud from segmented RGB-D.
    rgb: HxWx3 uint8 or float image
    depth: HxW depth map, raw units scaled by cfg.depth_scale
    mask: HxW, object mask
    """
    _require_open3d()
    if rgb.shape[:2] != depth.shape[:2] or depth.shape[:2] != mask.shape[:2]:
        raise ValueError(
            f"Shape mismatch: rgb={rgb.shape[:2]}, depth={depth.shape[:2]}, mask={mask.shape[:2]}"
        )
    H, W = depth.shape[:2]
    if H != intr.height or W != intr.width:
        raise ValueError(
            f"Image size ({W}x{H}) does not match intrinsics ({intr.width}x{intr.height})"
        )

    if rgb.dtype != np.uint8:
        rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        rgb_u8 = rgb

    if depth.dtype.kind in {"u", "i"}:
        depth_m = depth.astype(np.float32) / float(cfg.depth_scale)
    else:
        depth_m = depth.astype(np.float32)

    mask_u8 = erode_mask(mask, cfg.erode_kernel, cfg.erode_iters)
    obj_mask = mask_u8 > 0

    valid_depth = np.isfinite(depth_m) & (depth_m > cfg.depth_min_m) & (depth_m < cfg.depth_max_m)
    valid = obj_mask & valid_depth

    if cfg.remove_depth_edge_points:
        depth_edges = compute_depth_edge_mask(depth_m, valid_depth, cfg.depth_edge_thresh_m)
        valid = valid & (~depth_edges)

    v_coords, u_coords = np.where(valid)
    if len(u_coords) == 0:
        mask_px = int(obj_mask.sum())
        valid_depth_px = int(valid_depth.sum())
        overlap_px = int((obj_mask & valid_depth).sum())
        d_in_mask = depth_m[obj_mask] if mask_px > 0 else np.array([])
        d_stats = ""
        if len(d_in_mask) > 0:
            finite_d = d_in_mask[np.isfinite(d_in_mask)]
            if len(finite_d) > 0:
                d_stats = f", depth_in_mask: min={finite_d.min():.4f} max={finite_d.max():.4f} mean={finite_d.mean():.4f}"
            else:
                d_stats = ", depth_in_mask: all NaN/inf"
        raise ValueError(
            f"No valid object points after mask/depth filtering. "
            f"mask_pixels={mask_px}, valid_depth_pixels={valid_depth_px}, "
            f"overlap={overlap_px}, depth_range=[{cfg.depth_min_m}, {cfg.depth_max_m}]{d_stats}"
        )

    z = depth_m[v_coords, u_coords]
    x = (u_coords.astype(np.float64) - intr.cx) * z / intr.fx
    y = (v_coords.astype(np.float64) - intr.cy) * z / intr.fy
    xyz = np.stack([x, y, z], axis=1)

    colors = rgb_u8[v_coords, u_coords].astype(np.float64) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


# -------------------------
# Point cloud cleanup
# -------------------------
def _iqr_filter(pcd: o3d.geometry.PointCloud, multiplier: float) -> o3d.geometry.PointCloud:
    """Remove outliers using per-axis IQR (interquartile range) filtering."""
    _require_open3d()
    pts = np.asarray(pcd.points)
    if len(pts) < 10:
        return pcd
    keep = np.ones(len(pts), dtype=bool)
    for axis in range(3):
        vals = pts[:, axis]
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        lo = q1 - multiplier * iqr
        hi = q3 + multiplier * iqr
        keep &= (vals >= lo) & (vals <= hi)
    if keep.sum() == 0:
        return pcd
    return pcd.select_by_index(np.nonzero(keep)[0])


def preprocess_pcd(pcd: o3d.geometry.PointCloud, cfg: RegistrationConfig) -> o3d.geometry.PointCloud:
    _require_open3d()
    if cfg.voxel_size > 0:
        pcd = pcd.voxel_down_sample(cfg.voxel_size)
        if len(pcd.points) == 0:
            raise ValueError("Point cloud became empty after voxel downsampling.")

    if cfg.iqr_multiplier > 0:
        pcd = _iqr_filter(pcd, cfg.iqr_multiplier)
        if len(pcd.points) == 0:
            raise ValueError("Point cloud became empty after IQR outlier removal.")

    if cfg.outlier_nb_neighbors > 0 and cfg.outlier_std_ratio > 0:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=cfg.outlier_nb_neighbors,
            std_ratio=cfg.outlier_std_ratio,
        )
        if len(pcd.points) == 0:
            raise ValueError("Point cloud became empty after statistical outlier removal.")

    if cfg.radius_outlier_radius > 0 and cfg.radius_outlier_min_neighbors > 0:
        pcd, _ = pcd.remove_radius_outlier(
            nb_points=cfg.radius_outlier_min_neighbors,
            radius=cfg.radius_outlier_radius,
        )
        if len(pcd.points) == 0:
            raise ValueError("Point cloud became empty after radius outlier removal.")

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.normal_radius,
            max_nn=cfg.normal_max_nn,
        )
    )
    pcd.normalize_normals()
    return pcd


# -------------------------
# PCA-based weak init
# -------------------------
def compute_pca_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    X = points - centroid
    cov = (X.T @ X) / max(len(points) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    # right-handed frame
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1.0
    return centroid, eigvecs


def pca_initial_transform(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
) -> np.ndarray:
    candidates = pca_initial_transform_candidates(src_pts, tgt_pts)
    return candidates[0][0]


def pca_initial_transform_candidates(
    src_pts: np.ndarray,
    tgt_pts: np.ndarray,
) -> List[Tuple[np.ndarray, float]]:
    c_s, R_s = compute_pca_frame(src_pts)
    c_t, R_t = compute_pca_frame(tgt_pts)

    # Because PCA axes can have sign ambiguity, try all sign flips that preserve handedness.
    candidates = []
    sign_options = [
        np.diag([1, 1, 1]),
        np.diag([1, -1, -1]),
        np.diag([-1, 1, -1]),
        np.diag([-1, -1, 1]),
    ]
    for S in sign_options:
        R = R_t @ S @ R_s.T
        t = c_t - R @ c_s
        T = make_transform(R, t)
        candidates.append(T)

    # Choose by mean nearest-neighbor distance (more robust for near-symmetric shapes)
    from scipy.spatial import cKDTree
    tgt_tree = cKDTree(tgt_pts)
    scored: List[Tuple[np.ndarray, float]] = []
    for T in candidates:
        src_t = transform_points(src_pts, T)
        dists, _ = tgt_tree.query(src_t, k=1)
        score = float(np.mean(dists))
        scored.append((T, score))
    scored.sort(key=lambda item: item[1])
    return scored


def _prepare_fpfh_downsample(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius: float,
    feature_radius: float,
) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    _require_open3d()
    down = pcd.voxel_down_sample(float(voxel_size))
    if len(down.points) == 0:
        raise ValueError("FPFH RANSAC downsample produced an empty cloud.")
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius),
            max_nn=30,
        )
    )
    down.normalize_normals()
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        down,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(feature_radius),
            max_nn=100,
        ),
    )
    return down, fpfh


def fpfh_ransac_initial_transform(
    src_pcd: o3d.geometry.PointCloud,
    tgt_pcd: o3d.geometry.PointCloud,
    cfg: RegistrationConfig,
) -> tuple[np.ndarray, Dict[str, float]]:
    """Global point-cloud initialization using FPFH feature RANSAC.

    This is intentionally used as an initialization candidate only. Partial
    object observations can make FPFH ambiguous, so downstream visibility,
    silhouette, cloud overlap, shape, and ICP scoring remain the final judge.
    """
    _require_open3d()
    if not cfg.use_fpfh_ransac_init:
        raise ValueError("FPFH RANSAC init is disabled.")
    if len(src_pcd.points) < 20 or len(tgt_pcd.points) < 20:
        raise ValueError("Need at least 20 points per cloud for FPFH RANSAC.")

    voxel = max(float(cfg.fpfh_ransac_voxel_size), float(cfg.voxel_size), 1e-4)
    normal_radius = max(voxel * float(cfg.fpfh_normal_radius_factor), voxel + 1e-5)
    feature_radius = max(voxel * float(cfg.fpfh_feature_radius_factor), normal_radius + 1e-5)
    max_corr = max(voxel * float(cfg.fpfh_ransac_max_corr_factor), voxel + 1e-5)

    src_down, src_feat = _prepare_fpfh_downsample(src_pcd, voxel, normal_radius, feature_radius)
    tgt_down, tgt_feat = _prepare_fpfh_downsample(tgt_pcd, voxel, normal_radius, feature_radius)
    if len(src_down.points) < 8 or len(tgt_down.points) < 8:
        raise ValueError("Too few downsampled points for FPFH RANSAC.")

    checkers = [
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.90),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(max_corr),
    ]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(
        int(cfg.fpfh_ransac_max_iter),
        float(cfg.fpfh_ransac_confidence),
    )
    try:
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_feat,
            tgt_feat,
            True,
            max_corr,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            checkers,
            criteria,
        )
    except TypeError:
        # Compatibility with older Open3D signatures without mutual_filter.
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            tgt_down,
            src_feat,
            tgt_feat,
            max_corr,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            checkers,
            criteria,
        )

    info = {
        "fitness": float(result.fitness),
        "inlier_rmse": float(result.inlier_rmse),
        "src_down_points": float(len(src_down.points)),
        "tgt_down_points": float(len(tgt_down.points)),
        "voxel_size": float(voxel),
        "max_corr_dist": float(max_corr),
    }
    if result.fitness < float(cfg.fpfh_ransac_min_fitness):
        raise ValueError(
            f"FPFH RANSAC fitness too low: {result.fitness:.4f} < {cfg.fpfh_ransac_min_fitness:.4f}"
        )
    return np.asarray(result.transformation, dtype=np.float64), info


# -------------------------
# Projection / visibility-aware scoring
# -------------------------
def project_points_to_image(points_cam: np.ndarray, intr: CameraIntrinsics) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = points_cam[:, 2]
    valid_z = z > 1e-6
    u = np.full(len(points_cam), -1.0, dtype=np.float64)
    v = np.full(len(points_cam), -1.0, dtype=np.float64)

    u[valid_z] = intr.fx * points_cam[valid_z, 0] / z[valid_z] + intr.cx
    v[valid_z] = intr.fy * points_cam[valid_z, 1] / z[valid_z] + intr.cy

    inside = (
        valid_z
        & (u >= 0)
        & (u < intr.width - 1)
        & (v >= 0)
        & (v < intr.height - 1)
    )
    return u, v, inside


def bilinear_sample_scalar(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = np.clip(u0 + 1, 0, w - 1)
    v1 = np.clip(v0 + 1, 0, h - 1)
    u0 = np.clip(u0, 0, w - 1)
    v0 = np.clip(v0, 0, h - 1)

    du = u - u0
    dv = v - v0

    Ia = image[v0, u0]
    Ib = image[v0, u1]
    Ic = image[v1, u0]
    Id = image[v1, u1]

    return (
        Ia * (1 - du) * (1 - dv)
        + Ib * du * (1 - dv)
        + Ic * (1 - du) * dv
        + Id * du * dv
    )


def bilinear_sample_rgb(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    channels = [bilinear_sample_scalar(image[..., c], u, v) for c in range(3)]
    return np.stack(channels, axis=1)


def normals_to_camera(normals_base: np.ndarray, T_cam_to_base: np.ndarray) -> np.ndarray:
    R_cam_to_base = T_cam_to_base[:3, :3]
    R_base_to_cam = R_cam_to_base.T
    return (R_base_to_cam @ normals_base.T).T


@dataclass
class _LinearShapePrior:
    enabled: bool
    center: np.ndarray
    axis: np.ndarray
    length: float
    radius: float
    strength: float


def _build_linear_shape_prior(points: np.ndarray, cfg: RegistrationConfig) -> _LinearShapePrior:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    disabled = _LinearShapePrior(
        enabled=False,
        center=np.zeros(3, dtype=np.float64),
        axis=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        length=0.0,
        radius=0.0,
        strength=0.0,
    )
    if len(pts) < cfg.axis_prior_min_points:
        return disabled

    center = np.median(pts, axis=0)
    X = pts - center
    cov = (X.T @ X) / max(len(pts) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-12)
    eigvecs = eigvecs[:, order]
    axis = eigvecs[:, 0]
    axis /= max(np.linalg.norm(axis), 1e-12)

    linear_ratio = float(eigvals[0] / max(eigvals[1], 1e-12))
    if linear_ratio <= cfg.axis_prior_min_ratio:
        return disabled

    proj = X @ axis
    lo, hi = np.percentile(proj, [5.0, 95.0])
    length = float(hi - lo)
    if length <= 1e-6:
        return disabled

    radial = np.linalg.norm(X - np.outer(proj, axis), axis=1)
    radius = float(np.percentile(radial, 75.0))

    r0 = max(cfg.axis_prior_min_ratio, 1.01)
    r1 = max(cfg.axis_prior_full_ratio, r0 + 1e-6)
    strength = float(np.clip((math.log(linear_ratio) - math.log(r0)) / (math.log(r1) - math.log(r0)), 0.0, 1.0))
    return _LinearShapePrior(
        enabled=True,
        center=center,
        axis=axis,
        length=length,
        radius=radius,
        strength=strength,
    )


def _compute_axis_prior_scores(
    T_srcbase_to_tgtbase: np.ndarray,
    cache: "_ScoringCache",
) -> Dict[str, float]:
    zero = {
        "axis_prior_weight": 0.0,
        "axis_alignment_score": 0.0,
        "axis_center_score": 0.0,
        "axis_length_score": 0.0,
        "axis_score": 0.0,
    }
    cfg = cache.cfg
    if not cfg.axis_prior_enable:
        return zero

    src_prior = cache.src_shape_prior
    tgt_prior = cache.tgt_shape_prior
    if not src_prior.enabled or not tgt_prior.enabled:
        return zero

    weight = float(cfg.axis_prior_max_weight * min(src_prior.strength, tgt_prior.strength))
    if weight <= 1e-6:
        return zero

    R = np.asarray(T_srcbase_to_tgtbase, dtype=np.float64)[:3, :3]
    t = np.asarray(T_srcbase_to_tgtbase, dtype=np.float64)[:3, 3]
    src_axis_t = R @ src_prior.axis
    src_axis_t /= max(np.linalg.norm(src_axis_t), 1e-12)
    axis_alignment = float(np.clip(abs(np.dot(src_axis_t, tgt_prior.axis)), 0.0, 1.0))

    src_center_t = R @ src_prior.center + t
    center_delta = src_center_t - tgt_prior.center
    perp_delta = center_delta - np.dot(center_delta, tgt_prior.axis) * tgt_prior.axis
    center_tol = max(
        cfg.axis_prior_center_tol_m,
        0.75 * (src_prior.radius + tgt_prior.radius),
        cfg.voxel_size * 2.0,
    )
    center_score = float(math.exp(-np.linalg.norm(perp_delta) / max(center_tol, 1e-6)))

    length_tol = max(
        cfg.axis_prior_length_tol_ratio * max(src_prior.length, tgt_prior.length, 1e-6),
        cfg.voxel_size * 3.0,
    )
    length_score = float(math.exp(-abs(src_prior.length - tgt_prior.length) / max(length_tol, 1e-6)))

    axis_score = float(0.55 * axis_alignment + 0.30 * center_score + 0.15 * length_score)
    return {
        "axis_prior_weight": weight,
        "axis_alignment_score": axis_alignment,
        "axis_center_score": center_score,
        "axis_length_score": length_score,
        "axis_score": axis_score,
    }


@dataclass
class _ScoringCache:
    """Precomputed invariants for the hypothesis search loop."""
    src_points: np.ndarray
    src_colors: np.ndarray
    src_normals: np.ndarray
    src_shape_prior: _LinearShapePrior
    tgt_shape_prior: _LinearShapePrior
    T_base_to_wrist: np.ndarray
    R_base_to_wrist: np.ndarray
    wrist_depth_f64: np.ndarray
    wrist_rgb_f64: np.ndarray
    wrist_intr: CameraIntrinsics
    cfg: RegistrationConfig

    @staticmethod
    def build(
        src_pcd_base: o3d.geometry.PointCloud,
        tgt_pcd_base: Optional[o3d.geometry.PointCloud],
        wrist_depth: np.ndarray,
        wrist_rgb: np.ndarray,
        wrist_intr: CameraIntrinsics,
        T_wrist_to_base: np.ndarray,
        cfg: RegistrationConfig,
    ) -> "_ScoringCache":
        T_base_to_wrist = invert_transform(T_wrist_to_base)
        wrist_depth_f64 = wrist_depth.astype(np.float64)
        if wrist_rgb.dtype != np.float64:
            wrist_rgb_f64 = wrist_rgb.astype(np.float64)
        else:
            wrist_rgb_f64 = wrist_rgb.copy()
        if wrist_rgb_f64.max() > 1.5:
            wrist_rgb_f64 = wrist_rgb_f64 / 255.0
        src_points = np.asarray(src_pcd_base.points)
        tgt_points = np.asarray(tgt_pcd_base.points) if tgt_pcd_base is not None else np.zeros((0, 3), dtype=np.float64)
        return _ScoringCache(
            src_points=src_points,
            src_colors=np.asarray(src_pcd_base.colors),
            src_normals=np.asarray(src_pcd_base.normals),
            src_shape_prior=_build_linear_shape_prior(src_points, cfg),
            tgt_shape_prior=_build_linear_shape_prior(tgt_points, cfg),
            T_base_to_wrist=T_base_to_wrist,
            R_base_to_wrist=T_base_to_wrist[:3, :3].copy(),
            wrist_depth_f64=wrist_depth_f64,
            wrist_rgb_f64=wrist_rgb_f64,
            wrist_intr=wrist_intr,
            cfg=cfg,
        )


def _score_with_cache(T_srcbase_to_tgtbase: np.ndarray, cache: _ScoringCache) -> Dict[str, float]:
    """Fast scoring using precomputed invariants."""
    _BAD = {
        "score": -1e9,
        "visible_count": 0.0,
        "depth_ok_ratio": 0.0,
        "color_score": 0.0,
        "normal_score": 0.0,
        "axis_prior_weight": 0.0,
        "axis_alignment_score": 0.0,
        "axis_center_score": 0.0,
        "axis_length_score": 0.0,
        "axis_score": 0.0,
    }
    cfg = cache.cfg

    pts_base = transform_points(cache.src_points, T_srcbase_to_tgtbase)
    nrm_base = (T_srcbase_to_tgtbase[:3, :3] @ cache.src_normals.T).T

    pts_wrist = transform_points(pts_base, cache.T_base_to_wrist)
    nrm_wrist = (cache.R_base_to_wrist @ nrm_base.T).T

    u, v, inside = project_points_to_image(pts_wrist, cache.wrist_intr)
    if inside.sum() < cfg.min_visible_points:
        return {**_BAD, "visible_count": float(inside.sum())}

    u_in, v_in = u[inside], v[inside]
    z_in = pts_wrist[inside, 2]
    n_in, c_in = nrm_wrist[inside], cache.src_colors[inside]

    sampled_depth = bilinear_sample_scalar(cache.wrist_depth_f64, u_in, v_in)
    valid_depth = np.isfinite(sampled_depth) & (sampled_depth > cfg.depth_min_m) & (sampled_depth < cfg.depth_max_m)
    if valid_depth.sum() < cfg.min_visible_points:
        return {**_BAD, "visible_count": float(valid_depth.sum())}

    u_in, v_in = u_in[valid_depth], v_in[valid_depth]
    z_in = z_in[valid_depth]
    n_in, c_in = n_in[valid_depth], c_in[valid_depth]
    sampled_depth = sampled_depth[valid_depth]

    depth_err = np.abs(z_in - sampled_depth)
    depth_ok = depth_err < cfg.depth_consistency_thresh_m
    if depth_ok.sum() < cfg.min_visible_points:
        return {**_BAD, "visible_count": float(depth_ok.sum()),
                "depth_ok_ratio": float(depth_ok.mean()) if len(depth_ok) > 0 else 0.0}

    depth_consistency = np.exp(-depth_err[depth_ok].mean() / max(cfg.depth_consistency_thresh_m, 1e-6))

    sampled_rgb = bilinear_sample_rgb(cache.wrist_rgb_f64, u_in[depth_ok], v_in[depth_ok])
    color_err = np.linalg.norm(sampled_rgb - c_in[depth_ok], axis=1)
    color_score = np.exp(-color_err.mean() / 0.20)

    n = n_in[depth_ok]
    n_norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(n_norm, 1e-8)
    normal_score = np.clip(np.mean(np.abs(n[:, 2])), 0.0, 1.0)

    base_score = (
        cfg.depth_consistency_weight * depth_consistency
        + cfg.color_consistency_weight * color_score
        + cfg.normal_consistency_weight * normal_score
    )
    axis_info = _compute_axis_prior_scores(T_srcbase_to_tgtbase, cache)
    weight = float(axis_info.get("axis_prior_weight", 0.0))
    score = (1.0 - weight) * base_score + weight * float(axis_info.get("axis_score", 0.0))
    return {
        "score": float(score),
        "visible_count": float(len(z_in)),
        "depth_ok_ratio": float(depth_ok.mean()),
        "color_score": float(color_score),
        "normal_score": float(normal_score),
        **axis_info,
    }


def visibility_aware_score(
    src_pcd_base: o3d.geometry.PointCloud,
    tgt_pcd_base: Optional[o3d.geometry.PointCloud],
    T_srcbase_to_tgtbase: np.ndarray,
    wrist_depth: np.ndarray,
    wrist_rgb: np.ndarray,
    wrist_intr: CameraIntrinsics,
    T_wrist_to_base: np.ndarray,
    cfg: RegistrationConfig,
) -> Dict[str, float]:
    """
    Score a candidate transform by projecting transformed source cloud into wrist image.
    Convenience wrapper; builds cache internally for single-shot use.
    """
    cache = _ScoringCache.build(src_pcd_base, tgt_pcd_base, wrist_depth, wrist_rgb, wrist_intr, T_wrist_to_base, cfg)
    return _score_with_cache(T_srcbase_to_tgtbase, cache)


# -------------------------
# Local hypothesis search
# -------------------------
def local_hypothesis_search(
    src_head_pcd_base: o3d.geometry.PointCloud,
    tgt_pcd_base: Optional[o3d.geometry.PointCloud],
    T_init: np.ndarray,
    wrist_depth_m: np.ndarray,
    wrist_rgb: np.ndarray,
    wrist_intr: CameraIntrinsics,
    T_wrist_to_base: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    cache = _ScoringCache.build(
        src_head_pcd_base, tgt_pcd_base, wrist_depth_m, wrist_rgb, wrist_intr, T_wrist_to_base, cfg,
    )

    # Rotation pivot = where the source centroid lands under T_init
    src_centroid = cache.src_points.mean(axis=0)
    pivot = (T_init[:3, :3] @ src_centroid + T_init[:3, 3]).astype(np.float64)

    best_T = T_init.copy()
    best_info: Dict[str, float] = {"score": -1e9}

    for rx in cfg.search_rx_deg:
        for ry in cfg.search_ry_deg:
            for rz in cfg.search_rz_deg:
                dR = euler_xyz_deg_to_R(rx, ry, rz)
                for tx in cfg.search_tx_m:
                    for ty in cfg.search_ty_m:
                        for tz in cfg.search_tz_m:
                            dt = np.array([tx, ty, tz], dtype=np.float64)
                            # Rotate around object pivot, not world origin
                            rotated_pivot = dR @ pivot
                            t_corrected = pivot - rotated_pivot + dt
                            dT = make_transform(dR, t_corrected)
                            T_candidate = dT @ T_init
                            info = _score_with_cache(T_candidate, cache)
                            if info["score"] > best_info["score"]:
                                best_info = info
                                best_T = T_candidate
    return best_T, best_info


# -------------------------
# ICP refinement
# -------------------------
def copy_pcd(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    _require_open3d()
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).copy())
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).copy())
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).copy())
    return out


def refine_with_gicp(
    src_base: o3d.geometry.PointCloud,
    tgt_base: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    _require_open3d()
    result = o3d.pipelines.registration.registration_generalized_icp(
        source=src_base,
        target=tgt_base,
        max_correspondence_distance=cfg.gicp_max_corr_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.gicp_max_iter),
    )
    return result.transformation, result


def refine_with_colored_icp(
    src_base: o3d.geometry.PointCloud,
    tgt_base: o3d.geometry.PointCloud,
    T_init: np.ndarray,
    cfg: RegistrationConfig,
) -> Tuple[np.ndarray, o3d.pipelines.registration.RegistrationResult]:
    _require_open3d()
    result = o3d.pipelines.registration.registration_colored_icp(
        source=src_base,
        target=tgt_base,
        max_correspondence_distance=cfg.colored_icp_max_corr_dist,
        init=T_init,
        estimation_method=o3d.pipelines.registration.TransformationEstimationForColoredICP(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=cfg.colored_icp_max_iter),
    )
    return result.transformation, result


# -------------------------
# Visualization
# -------------------------
def visualize_alignment(
    src_head_base: o3d.geometry.PointCloud,
    tgt_wrist_base: o3d.geometry.PointCloud,
    T_src_to_tgt: np.ndarray,
    window_name: str = "alignment",
) -> None:
    src_vis = copy_pcd(src_head_base)
    tgt_vis = copy_pcd(tgt_wrist_base)
    src_vis.paint_uniform_color([1.0, 0.2, 0.2])
    tgt_vis.paint_uniform_color([0.2, 0.8, 0.2])
    src_vis.transform(T_src_to_tgt)
    o3d.visualization.draw_geometries([src_vis, tgt_vis], window_name=window_name)


# -------------------------
# Main API
# -------------------------
def align_object_partial_rgbd(
    head_rgb: np.ndarray,
    head_depth: np.ndarray,
    head_mask: np.ndarray,
    wrist_rgb: np.ndarray,
    wrist_depth: np.ndarray,
    wrist_mask: np.ndarray,
    K_head: CameraIntrinsics,
    K_wrist: CameraIntrinsics,
    T_head_to_base: np.ndarray,
    T_wrist_to_ee: np.ndarray,
    T_ee_to_base: np.ndarray,
    cfg: Optional[RegistrationConfig] = None,
    debug: bool = False,
) -> Dict[str, object]:
    """
    Returns a dict with:
      - T_headobjbase_to_wristobjbase: 4x4 transform mapping the head partial object cloud in base frame
        into the wrist partial object pose in base frame.
      - T_headcamobj_to_wristcamobj: 4x4 transform directly mapping head-camera object points
        into wrist-camera object points.
      - pcd_head_base, pcd_wrist_base: processed point clouds in base frame
      - intermediate scores/results
    """
    if cfg is None:
        cfg = RegistrationConfig()

    T_head_to_base = to_homogeneous(T_head_to_base)
    T_wrist_to_ee = to_homogeneous(T_wrist_to_ee)
    T_ee_to_base = to_homogeneous(T_ee_to_base)
    T_wrist_to_base = T_ee_to_base @ T_wrist_to_ee

    # 1) RGB-D -> object-only point clouds in each camera frame
    pcd_head_cam = rgbd_to_pointcloud(head_rgb, head_depth, head_mask, K_head, cfg)
    pcd_wrist_cam = rgbd_to_pointcloud(wrist_rgb, wrist_depth, wrist_mask, K_wrist, cfg)

    # 2) cleanup + normals
    pcd_head_cam = preprocess_pcd(pcd_head_cam, cfg)
    pcd_wrist_cam = preprocess_pcd(pcd_wrist_cam, cfg)

    # 3) move into base frame
    pcd_head_base = copy_pcd(pcd_head_cam)
    pcd_head_base.transform(T_head_to_base)

    pcd_wrist_base = copy_pcd(pcd_wrist_cam)
    pcd_wrist_base.transform(T_wrist_to_base)

    # 4) initial transform (maps head-base object cloud -> wrist-base object cloud)
    src_pts = np.asarray(pcd_head_base.points)
    tgt_pts = np.asarray(pcd_wrist_base.points)

    if cfg.use_pca_init:
        T_init = pca_initial_transform(src_pts, tgt_pts)
    else:
        # centroid-only fallback
        t = tgt_pts.mean(axis=0) - src_pts.mean(axis=0)
        T_init = make_transform(np.eye(3), t)

    # wrist depth -> meters for scoring
    if wrist_depth.dtype.kind in {"u", "i"}:
        wrist_depth_m = wrist_depth.astype(np.float32) / float(cfg.depth_scale)
    else:
        wrist_depth_m = wrist_depth.astype(np.float32)

    # 5) local search around init
    T_search, search_info = local_hypothesis_search(
        src_head_pcd_base=pcd_head_base,
        tgt_pcd_base=pcd_wrist_base,
        T_init=T_init,
        wrist_depth_m=wrist_depth_m,
        wrist_rgb=wrist_rgb,
        wrist_intr=K_wrist,
        T_wrist_to_base=T_wrist_to_base,
        cfg=cfg,
    )

    # 6) GICP refinement in base frame
    T_gicp, gicp_result = refine_with_gicp(
        src_base=pcd_head_base,
        tgt_base=pcd_wrist_base,
        T_init=T_search,
        cfg=cfg,
    )

    T_final = T_gicp
    colored_result = None

    # 7) optional colored ICP refinement
    if cfg.use_colored_icp:
        # colored ICP requires normals too; we already have them
        T_col, colored_result = refine_with_colored_icp(
            src_base=pcd_head_base,
            tgt_base=pcd_wrist_base,
            T_init=T_gicp,
            cfg=cfg,
        )
        T_final = T_col

    # direct head-cam -> wrist-cam transform
    T_headcam_to_wristcam = invert_transform(T_wrist_to_base) @ T_final @ T_head_to_base

    result = {
        "T_headobjbase_to_wristobjbase": T_final,
        "T_headcamobj_to_wristcamobj": T_headcam_to_wristcam,
        "pcd_head_base": pcd_head_base,
        "pcd_wrist_base": pcd_wrist_base,
        "T_init": T_init,
        "T_after_search": T_search,
        "search_info": search_info,
        "gicp_fitness": float(gicp_result.fitness),
        "gicp_inlier_rmse": float(gicp_result.inlier_rmse),
        "colored_fitness": None if colored_result is None else float(colored_result.fitness),
        "colored_inlier_rmse": None if colored_result is None else float(colored_result.inlier_rmse),
    }

    if debug:
        print("===== Registration Debug =====")
        print("search score:", search_info)
        print("gicp fitness:", result["gicp_fitness"])
        print("gicp rmse:", result["gicp_inlier_rmse"])
        if colored_result is not None:
            print("colored fitness:", result["colored_fitness"])
            print("colored rmse:", result["colored_inlier_rmse"])
        visualize_alignment(pcd_head_base, pcd_wrist_base, T_final, window_name="final alignment")

    return result


# -------------------------
# Example usage
# -------------------------
def example_usage():
    """
    Replace the placeholders below with your real data.
    """
    # Example placeholders:
    head_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    head_depth = np.zeros((480, 640), dtype=np.uint16)
    head_mask = np.zeros((480, 640), dtype=np.uint8)

    wrist_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    wrist_depth = np.zeros((480, 640), dtype=np.uint16)
    wrist_mask = np.zeros((480, 640), dtype=np.uint8)

    K_head = CameraIntrinsics(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480
    )
    K_wrist = CameraIntrinsics(
        fx=600.0, fy=600.0, cx=320.0, cy=240.0, width=640, height=480
    )

    # 4x4 extrinsics
    T_head_to_base = np.eye(4)
    T_wrist_to_ee = np.eye(4)
    T_ee_to_base = np.eye(4)

    cfg = RegistrationConfig(
        depth_scale=1000.0,
        voxel_size=0.003,
        gicp_max_corr_dist=0.01,
        use_colored_icp=True,
    )

    result = align_object_partial_rgbd(
        head_rgb=head_rgb,
        head_depth=head_depth,
        head_mask=head_mask,
        wrist_rgb=wrist_rgb,
        wrist_depth=wrist_depth,
        wrist_mask=wrist_mask,
        K_head=K_head,
        K_wrist=K_wrist,
        T_head_to_base=T_head_to_base,
        T_wrist_to_ee=T_wrist_to_ee,
        T_ee_to_base=T_ee_to_base,
        cfg=cfg,
        debug=True,
    )

    print("Final T (head object in base -> wrist object in base):")
    print(result["T_headobjbase_to_wristobjbase"])
    print("Direct T (head camera object -> wrist camera object):")
    print(result["T_headcamobj_to_wristcamobj"])


if __name__ == "__main__":
    example_usage()
