from __future__ import annotations

import argparse
import base64
import io
import itertools
import os
from dataclasses import dataclass, is_dataclass
from io import BytesIO
from typing import Optional, Tuple, Dict, Any, Union

import cv2
import numpy as np
try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]
import requests
try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]
import time

SAM3_SERVER_URL = os.environ.get("SAM3_SERVER_URL", "http://101.132.143.105:5081")
DEFAULT_FX = 439.29821777
DEFAULT_FY = 439.29821777
DEFAULT_CX = 640.79266357
DEFAULT_CY = 346.89926147
DEFAULT_BASELINE = 0.080010292006238


def _require_open3d() -> None:
    if o3d is None:
        raise ImportError("open3d is required for point-cloud visualization helpers.")

def segment_with_sam3(
    image_path: Optional[str] = None,
    image_base64: Optional[str] = None,
    text_prompt: Optional[str] = None,
    box_prompt: Optional[list[float]] = None,
    point_prompts: Optional[list[list[float]]] = None,
    point_labels: Optional[list[int]] = None,
    server_url: Optional[str] = None,
    timeout: float = 60.0,
    verbose: bool = False,
) -> Optional[Dict[str, Any]]:
    """调用 SAM3 服务进行分割，返回原始 JSON 结果。"""
    server_url = server_url or SAM3_SERVER_URL

    if image_path is None and image_base64 is None:
        raise ValueError("必须提供 image_path 或 image_base64")

    if image_base64 is None:
        with open(image_path, "rb") as f:
            image_base64 = base64.b64encode(f.read()).decode("utf-8")

    payload: Dict[str, Any] = {"image": image_base64}
    if text_prompt:
        payload["text_prompt"] = text_prompt
    if box_prompt is not None:
        payload["box_prompt"] = box_prompt
    if point_prompts is not None:
        payload["point_prompts"] = point_prompts
    if point_labels is not None:
        payload["point_labels"] = point_labels

    if not any(key in payload for key in ("text_prompt", "box_prompt", "point_prompts")):
        raise ValueError("必须至少提供 text_prompt、box_prompt 或 point_prompts 之一")
    if ("point_prompts" in payload) != ("point_labels" in payload):
        raise ValueError("point_prompts 和 point_labels 必须同时提供")

    if verbose:
        print(f"SAM3 request -> {server_url}/segment")

    try:
        response = requests.post(
            f"{server_url}/segment",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=timeout,
        )
    except requests.exceptions.Timeout:
        if verbose:
            print("SAM3 request timeout")
        return None
    except requests.exceptions.ConnectionError:
        if verbose:
            print(f"SAM3 connection failed: {server_url}")
        return None
    except Exception as exc:
        if verbose:
            print(f"SAM3 request failed: {exc}")
        return None

    if response.status_code != 200:
        if verbose:
            print(f"SAM3 bad status: {response.status_code} {response.text}")
        return None
    return response.json()

def request_sam3_mask(
    image_path: str,
    text_prompt: Optional[str] = None,
    box_prompt: Optional[list[float]] = None,
    point_prompts: Optional[list[list[float]]] = None,
    point_labels: Optional[list[int]] = None,
    server_url: Optional[str] = None,
    timeout: float = 60.0,
    verbose: bool = False,
) -> np.ndarray:
    """调用 SAM3 并返回二值 mask（uint8, 0/255）。"""
    if Image is None:
        raise ImportError("Pillow is required for request_sam3_mask")
    seg_result = segment_with_sam3(
        image_path=image_path,
        text_prompt=text_prompt,
        box_prompt=box_prompt,
        point_prompts=point_prompts,
        point_labels=point_labels,
        server_url=server_url,
        timeout=timeout,
        verbose=verbose,
    )
    if not seg_result or not seg_result.get("success"):
        error_msg = seg_result.get("error", "未知错误") if seg_result else "请求失败"
        raise ValueError(f"SAM3分割失败: {error_msg}")

    detections = seg_result.get("detections", [])
    if len(detections) == 0:
        raise ValueError("未检测到任何目标")

    mask_data = base64.b64decode(detections[0]["mask"])
    mask_img = Image.open(BytesIO(mask_data))
    return np.where(np.array(mask_img) > 0, 255, 0).astype(np.uint8)

@dataclass
class CuboidFitResult:
    center: np.ndarray          # (3,) 相机系下长方体中心
    rotation: np.ndarray        # (3,3), 列=[盒体局部 x, 盒体局部 y, 盒体局部 z]（相机系）
    size: np.ndarray            # (3,) = [size_x, size_y, size_z]，分别沿 cuboid_x/y/z
    min_local: np.ndarray       # (3,)
    max_local: np.ndarray       # (3,)
    corners_world: np.ndarray   # (8,3)；字段名保留兼容，但语义为相机系
    face_poses_wxyz: Dict[str, Dict[str, np.ndarray]]  # 六个面的位姿（相机系）
    residual: float

@dataclass
class CylinderFitResult:
    center: np.ndarray          # (3,) 相机系中心
    axis: np.ndarray            # (3,) 柱轴方向（单位向量）
    semi_major: float           # 截面椭圆长半轴
    semi_minor: float           # 截面椭圆短半轴
    height: float
    top_center: np.ndarray      # (3,)
    bottom_center: np.ndarray   # (3,)
    rotation: np.ndarray        # (3,3) 列=[长轴方向, 短轴方向, 柱轴]（相机系）
    residual: float

@dataclass
class FrustumFitResult:
    center: np.ndarray          # (3,) 相机系中心（截锥高度中点）
    small_end_center: np.ndarray  # (3,) 相机系小端中心
    large_end_center: np.ndarray  # (3,) 相机系大端中心
    axis: np.ndarray              # (3,) 从小端指向大端
    small_end_major: float        # 小端椭圆长半轴
    small_end_minor: float        # 小端椭圆短半轴
    large_end_major: float        # 大端椭圆长半轴
    large_end_minor: float        # 大端椭圆短半轴
    height: float
    rotation: np.ndarray        # (3,3) 列=[长轴方向, 短轴方向, 截锥轴]（相机系）
    pose_type: Optional[str]    # 纯几何相机系模式下默认为 None
    residual: float

@dataclass
class BowlFitResult:
    vertex: np.ndarray          # (3,) 抛物面顶点（相机系）
    rim_center: np.ndarray      # (3,) 碗口中心（相机系）
    axis: np.ndarray            # (3,) 从顶点指向碗口中心
    rotation: np.ndarray        # (3,3) 列=[长轴方向, 短轴方向, 碗轴]（相机系）
    rim_major: float            # 碗口长半轴
    rim_minor: float            # 碗口短半轴
    depth: float                # 顶点到碗口中心的深度
    pose_type: Optional[str]    # 纯几何相机系模式下默认为 None
    residual: float

@dataclass
class PlateFitResult:
    vertex: np.ndarray          # (3,) 平底中心（相机系）
    rim_center: np.ndarray      # (3,) 盘沿中心（相机系）
    axis: np.ndarray            # (3,) 从顶点指向盘沿中心
    rotation: np.ndarray        # (3,3) 列=[长轴方向, 短轴方向, 盘轴]（相机系）
    bottom_major: float         # 平底长半轴
    bottom_minor: float         # 平底短半轴
    rim_major: float            # 盘沿长半轴
    rim_minor: float            # 盘沿短半轴
    depth: float                # 盘心到盘沿中心的深度
    pose_type: Optional[str]    # 纯几何相机系模式下默认为 None
    bottom_flatness: float      # 平底区域高度起伏（std）
    residual: float

def ensure_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N,3), got {points.shape}")
    if len(points) < 20:
        raise ValueError("Need at least 20 points.")
    return points

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("Zero norm vector.")
    return v / n

def make_right_handed(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).copy()
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1.0
    return R

def rotation_matrix_to_quaternion_wxyz(R: np.ndarray) -> np.ndarray:
    """3x3旋转矩阵 -> 四元数 [w, x, y, z]。"""
    R = np.asarray(R, dtype=np.float64)
    t = float(np.trace(R))
    if t > 0.0:
        s = 2.0 * np.sqrt(t + 1.0)
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.array([w, x, y, z], dtype=np.float64)
    q /= max(np.linalg.norm(q), 1e-12)
    if q[0] < 0:
        q = -q
    return q

def quaternion_wxyz_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """四元数 [w, x, y, z] -> 3x3 旋转矩阵。"""
    q = np.asarray(q, dtype=np.float64).ravel()
    if q.shape[0] != 4:
        raise ValueError(f"quaternion must have 4 elements, got shape {q.shape}")
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Zero norm quaternion.")
    w, x, y, z = q / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)

def _frame_from_normal_and_xhint(normal: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    """构造面坐标系：z轴为外法向，x/y轴沿盒体边方向。"""
    z_axis = normalize(normal)
    x_axis = np.asarray(x_hint, dtype=np.float64)
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < 1e-8:
        if abs(z_axis[0]) < 0.9:
            x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        else:
            x_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    return make_right_handed(np.column_stack([x_axis, y_axis, z_axis]))

def cuboid_face_quaternions_wxyz(rotation: np.ndarray) -> Dict[str, np.ndarray]:
    """返回6个面的世界系四元数，面坐标系z轴为外法向。"""
    rotation = make_right_handed(np.asarray(rotation, dtype=np.float64))
    x_axis = rotation[:, 0]
    y_axis = rotation[:, 1]
    z_axis = rotation[:, 2]

    face_frames = {
        "x_plus": _frame_from_normal_and_xhint(+x_axis, y_axis),
        "x_minus": _frame_from_normal_and_xhint(-x_axis, y_axis),
        "y_plus": _frame_from_normal_and_xhint(+y_axis, z_axis),
        "y_minus": _frame_from_normal_and_xhint(-y_axis, z_axis),
        "z_plus": _frame_from_normal_and_xhint(+z_axis, x_axis),
        "z_minus": _frame_from_normal_and_xhint(-z_axis, x_axis),
    }
    return {
        name: rotation_matrix_to_quaternion_wxyz(face_R)
        for name, face_R in face_frames.items()
    }

def cuboid_face_centers_world(
    center: np.ndarray,
    rotation: np.ndarray,
    size: np.ndarray,
) -> Dict[str, np.ndarray]:
    center = np.asarray(center, dtype=np.float64)
    rotation = make_right_handed(np.asarray(rotation, dtype=np.float64))
    size = np.asarray(size, dtype=np.float64)
    half = 0.5 * size
    face_centers_local = {
        "x_plus":  np.array([+half[0], 0.0, 0.0], dtype=np.float64),
        "x_minus": np.array([-half[0], 0.0, 0.0], dtype=np.float64),
        "y_plus":  np.array([0.0, +half[1], 0.0], dtype=np.float64),
        "y_minus": np.array([0.0, -half[1], 0.0], dtype=np.float64),
        "z_plus":  np.array([0.0, 0.0, +half[2]], dtype=np.float64),
        "z_minus": np.array([0.0, 0.0, -half[2]], dtype=np.float64),
    }
    return {
        name: transform_local_to_world(local_center[None], center, rotation)[0]
        for name, local_center in face_centers_local.items()
    }

def cuboid_face_poses_wxyz(
    center: np.ndarray,
    rotation: np.ndarray,
    size: np.ndarray,
) -> Dict[str, Dict[str, np.ndarray]]:
    face_centers = cuboid_face_centers_world(center, rotation, size)
    face_quats = cuboid_face_quaternions_wxyz(rotation)
    return {
        face_name: {
            "position": face_centers[face_name],
            "quaternion_wxyz": face_quats[face_name],
        }
        for face_name in face_quats
    }

def transform_world_to_local(points: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # rotation columns are local axes expressed in world frame
    # local = R^T (p - center)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return (points - center) @ rotation

def transform_local_to_world(points_local: np.ndarray, center: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    # world = center + R local
    return points_local @ rotation.T + center

def pca_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    centroid = points.mean(axis=0)
    X = points - centroid
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    eigvecs = make_right_handed(eigvecs)
    return centroid, eigvecs, eigvals

def orthonormal_basis_from_z(z_axis: np.ndarray, x_hint: Optional[np.ndarray] = None) -> np.ndarray:
    z = normalize(z_axis)

    if x_hint is None:
        if abs(z[0]) < 0.9:
            x_hint = np.array([1.0, 0.0, 0.0])
        else:
            x_hint = np.array([0.0, 1.0, 0.0])
    else:
        x_hint = np.asarray(x_hint, dtype=np.float64)

    x = x_hint - np.dot(x_hint, z) * z
    if np.linalg.norm(x) < 1e-8:
        if abs(z[0]) < 0.9:
            x_hint = np.array([1.0, 0.0, 0.0])
        else:
            x_hint = np.array([0.0, 1.0, 0.0])
        x = x_hint - np.dot(x_hint, z) * z

    x = normalize(x)
    y = np.cross(z, x)
    y = normalize(y)
    R = np.stack([x, y, z], axis=1)
    return make_right_handed(R)

def angle_axis_rotation(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c
    R = np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C],
    ], dtype=np.float64)
    return R

def rotation_about_axis(axis: np.ndarray, theta: float) -> np.ndarray:
    return angle_axis_rotation(axis, theta)

def numpy_to_pcd(points: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return pcd

def pcd_to_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    return np.asarray(pcd.points, dtype=np.float64)

def keep_largest_cluster(pcd: o3d.geometry.PointCloud, eps: float = 0.02, min_points: int = 30,) -> o3d.geometry.PointCloud:
    if len(pcd.points) == 0:
        return pcd

    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
    if labels.size == 0 or labels.max() < 0:
        return pcd

    valid = labels >= 0
    if valid.sum() == 0:
        return pcd

    unique_labels, counts = np.unique(labels[valid], return_counts=True)
    best_label = unique_labels[np.argmax(counts)]
    idx = np.where(labels == best_label)[0]
    return pcd.select_by_index(idx)

def remove_point_outliers(
    pcd: o3d.geometry.PointCloud,
    statistical_nb_neighbors: Optional[int] = None,
    statistical_std_ratio: float = 1.8,
    radius_nb_points: Optional[int] = None,
    radius_scale: float = 4.0,
    min_points_keep: int = 20,
) -> Tuple[o3d.geometry.PointCloud, Dict[str, Any]]:
    """两阶段离群点过滤：统计离群点 + 基于自适应半径的半径离群点。"""
    initial_count = int(len(pcd.points))
    stats: Dict[str, Any] = {
        "initial_points": initial_count,
        "after_statistical": initial_count,
        "after_radius": initial_count,
        "final_points": initial_count,
    }
    if initial_count < max(min_points_keep * 2, 40):
        return pcd, stats

    filtered = pcd
    n_points = len(filtered.points)
    nb_neighbors = (
        int(np.clip(np.sqrt(n_points), 12, 32))
        if statistical_nb_neighbors is None else int(statistical_nb_neighbors)
    )

    try:
        statistical_pcd, inlier_idx = filtered.remove_statistical_outlier(
            nb_neighbors=max(nb_neighbors, 4),
            std_ratio=float(statistical_std_ratio),
        )
        if len(inlier_idx) >= min_points_keep:
            filtered = statistical_pcd
        stats["after_statistical"] = int(len(filtered.points))
    except Exception as exc:
        stats["statistical_error"] = f"{type(exc).__name__}: {exc}"

    try:
        nn_dists = np.asarray(filtered.compute_nearest_neighbor_distance(), dtype=np.float64)
        nn_dists = nn_dists[np.isfinite(nn_dists) & (nn_dists > 1e-8)]
        if len(nn_dists) > 0:
            median_nn = float(np.median(nn_dists))
            radius = max(radius_scale * median_nn, 0.003)
            radius_min_points = (
                int(np.clip(nb_neighbors // 3, 4, 10))
                if radius_nb_points is None else int(radius_nb_points)
            )
            radius_pcd, inlier_idx = filtered.remove_radius_outlier(
                nb_points=max(radius_min_points, 2),
                radius=radius,
            )
            stats["radius"] = radius
            stats["radius_nb_points"] = int(max(radius_min_points, 2))
            if len(inlier_idx) >= min_points_keep:
                filtered = radius_pcd
        stats["after_radius"] = int(len(filtered.points))
    except Exception as exc:
        stats["radius_error"] = f"{type(exc).__name__}: {exc}"

    stats["final_points"] = int(len(filtered.points))
    return filtered, stats

def point_to_aabb_surface_distance(points_local: np.ndarray, min_local: np.ndarray, max_local: np.ndarray) -> np.ndarray:
    d_out = np.maximum(np.maximum(min_local - points_local, 0.0), points_local - max_local)
    dist_out = np.linalg.norm(d_out, axis=1)

    inside = np.all((points_local >= min_local) & (points_local <= max_local), axis=1)
    d_to_min = points_local - min_local
    d_to_max = max_local - points_local
    dist_in = np.minimum(d_to_min, d_to_max).min(axis=1)

    dist = dist_out.copy()
    dist[inside] = dist_in[inside]
    return dist

def cuboid_corners_from_bounds(min_local: np.ndarray, max_local: np.ndarray) -> np.ndarray:
    x0, y0, z0 = min_local
    x1, y1, z1 = max_local
    corners = np.array([
        [x0, y0, z0],
        [x0, y0, z1],
        [x0, y1, z0],
        [x0, y1, z1],
        [x1, y0, z0],
        [x1, y0, z1],
        [x1, y1, z0],
        [x1, y1, z1],
    ], dtype=np.float64)
    return corners

def fit_ellipse_2d(xy: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Fitzgibbon direct least-squares ellipse fit.

    Returns (cx, cy, semi_major, semi_minor, angle_rad).
    angle_rad: 长轴从局部 x 轴的旋转角。
    若拟合退化则退回圆拟合。
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = len(xy)
    if n < 6:
        r = float(np.sqrt(np.mean(np.sum((xy - xy.mean(axis=0))**2, axis=1))))
        r = max(r, 1e-6)
        return float(xy[:, 0].mean()), float(xy[:, 1].mean()), r, r, 0.0

    mx, my = xy.mean(axis=0)
    x, y = xy[:, 0] - mx, xy[:, 1] - my

    D = np.column_stack([x**2, x * y, y**2, x, y, np.ones(n)])
    S = D.T @ D
    S1, S2, S3 = S[:3, :3], S[:3, 3:], S[3:, 3:]
    C1 = np.array([[0, 0, 2], [0, -1, 0], [2, 0, 0]], dtype=np.float64)

    try:
        T = -np.linalg.solve(S3, S2.T)
        M = np.linalg.solve(C1, S1 + S2 @ T)
        eigvals, eigvecs = np.linalg.eig(M)
        eigvals, eigvecs = eigvals.real, eigvecs.real
        cond = 4 * eigvecs[0] * eigvecs[2] - eigvecs[1] ** 2
        valid = np.where(cond > 0)[0]
        if len(valid) == 0:
            raise ValueError("no valid ellipse eigenvector")
        idx = valid[np.argmin(np.abs(eigvals[valid]))]
        a1 = eigvecs[:, idx]
        a2 = T @ a1
        A, B, C_, D_, E, F = np.concatenate([a1, a2])
    except Exception:
        r = float(np.sqrt(np.mean(x**2 + y**2)))
        return mx, my, max(r, 1e-6), max(r, 1e-6), 0.0

    denom = B**2 - 4 * A * C_
    if abs(denom) < 1e-15:
        r = float(np.sqrt(np.mean(x**2 + y**2)))
        return mx, my, max(r, 1e-6), max(r, 1e-6), 0.0

    cx_l = (2 * C_ * D_ - B * E) / denom
    cy_l = (2 * A * E - B * D_) / denom

    num = 2 * (A * E**2 + C_ * D_**2 - B * D_ * E + denom * F)
    s_sum = A + C_
    s_diff = np.sqrt(max((A - C_)**2 + B**2, 0.0))
    d1 = denom * (s_sum + s_diff)
    d2 = denom * (s_sum - s_diff)

    if abs(d1) < 1e-15 or abs(d2) < 1e-15:
        r = float(np.sqrt(np.mean(x**2 + y**2)))
        return mx, my, max(r, 1e-6), max(r, 1e-6), 0.0

    a_sq, b_sq = -num / d1, -num / d2
    if a_sq <= 0 or b_sq <= 0:
        r = float(np.sqrt(np.mean(x**2 + y**2)))
        return mx, my, max(r, 1e-6), max(r, 1e-6), 0.0

    sa, sb = np.sqrt(a_sq), np.sqrt(b_sq)
    angle = 0.5 * np.arctan2(B, A - C_)
    if sa < sb:
        sa, sb = sb, sa
        angle += np.pi / 2

    return float(cx_l + mx), float(cy_l + my), float(sa), float(sb), float(angle)


def _ellipse_radial_distance(u: np.ndarray, v: np.ndarray, a: float, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (到椭圆边界的近似距离, 是否在椭圆内部)。"""
    theta = np.arctan2(v * a, u * b)
    eu, ev = a * np.cos(theta), b * np.sin(theta)
    dist = np.sqrt((u - eu)**2 + (v - ev)**2)
    inside = (u / a)**2 + (v / b)**2 <= 1.0
    return dist, inside


def point_to_ellip_cylinder_distance(
    points: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    semi_major: float,
    semi_minor: float,
    height: float,
) -> np.ndarray:
    """点到椭圆柱表面的近似距离。rotation 列=[长轴, 短轴, 柱轴]。"""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        local = (points - center) @ rotation
    u, v, w = local[:, 0], local[:, 1], local[:, 2]
    half_h = height / 2.0

    radial_dist, inside_r = _ellipse_radial_distance(u, v, semi_major, semi_minor)
    inside_z = np.abs(w) <= half_h
    axial_excess = np.maximum(np.abs(w) - half_h, 0.0)

    dist = np.zeros(len(points), dtype=np.float64)
    m = (~inside_r) & inside_z
    dist[m] = radial_dist[m]
    m = inside_r & (~inside_z)
    dist[m] = axial_excess[m]
    m = (~inside_r) & (~inside_z)
    dist[m] = np.sqrt(radial_dist[m]**2 + axial_excess[m]**2)
    m = inside_r & inside_z
    dist[m] = np.minimum(radial_dist[m], half_h - np.abs(w[m]))
    return dist

def point_to_ellip_frustum_distance(
    points: np.ndarray,
    small_end_center: np.ndarray,
    rotation: np.ndarray,
    small_end_major: float,
    small_end_minor: float,
    large_end_major: float,
    large_end_minor: float,
    height: float,
) -> np.ndarray:
    """点到椭圆截锥表面的近似距离。rotation 列=[长轴, 短轴, 杯轴]。"""
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        local = (points - small_end_center) @ rotation
    u, v, w = local[:, 0], local[:, 1], local[:, 2]

    safe_height = max(float(height), 1e-8)
    t = np.clip(w / safe_height, 0.0, 1.0)
    semi_major = small_end_major + (large_end_major - small_end_major) * t
    semi_minor = small_end_minor + (large_end_minor - small_end_minor) * t

    radial_dist, inside_r = _ellipse_radial_distance(u, v, semi_major, semi_minor)
    inside_z = (w >= 0.0) & (w <= height)
    axial_excess = np.maximum(np.maximum(-w, 0.0), w - height)

    dist = np.zeros(len(points), dtype=np.float64)
    m = (~inside_r) & inside_z
    dist[m] = radial_dist[m]
    m = inside_r & (~inside_z)
    dist[m] = axial_excess[m]
    m = (~inside_r) & (~inside_z)
    dist[m] = np.sqrt(radial_dist[m]**2 + axial_excess[m]**2)
    m = inside_r & inside_z
    dist[m] = np.minimum(radial_dist[m], np.minimum(w[m], height - w[m]))
    return dist

def _make_frame_from_major_axis(major_dir: np.ndarray, axis: np.ndarray) -> np.ndarray:
    axis = normalize(axis)
    major_dir = np.asarray(major_dir, dtype=np.float64)
    major_dir = major_dir - np.dot(major_dir, axis) * axis
    major_dir = normalize(major_dir)
    minor_dir = normalize(np.cross(axis, major_dir))
    return np.column_stack([major_dir, minor_dir, axis])

def _stable_inplane_major_direction(
    axis: np.ndarray,
    reference_dirs: Tuple[np.ndarray, ...] = (),
) -> np.ndarray:
    axis = normalize(axis)
    candidates = [np.asarray(ref, dtype=np.float64) for ref in reference_dirs]
    candidates.extend([
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    ])

    for candidate in candidates:
        projected = candidate - np.dot(candidate, axis) * axis
        norm = np.linalg.norm(projected)
        if norm <= 1e-6:
            continue
        major_dir = projected / norm
        pivot = int(np.argmax(np.abs(major_dir)))
        if major_dir[pivot] < 0.0:
            major_dir = -major_dir
        return major_dir

    fallback = orthonormal_basis_from_z(axis)[:, 0]
    pivot = int(np.argmax(np.abs(fallback)))
    if fallback[pivot] < 0.0:
        fallback = -fallback
    return fallback

def _is_nearly_round_cross_section(
    semi_major: float,
    semi_minor: float,
    threshold: float = 1.12,
) -> bool:
    return float(semi_major / max(semi_minor, 1e-8)) <= threshold

def _dish_depth_ratio(depth: float, rim_minor: float) -> float:
    return float(depth / max(rim_minor, 1e-8))

def _append_unique_axis(candidates: list[np.ndarray], axis: np.ndarray, min_dot: float = 0.999) -> None:
    axis = normalize(axis)
    for existing in candidates:
        if float(np.dot(existing, axis)) >= min_dot:
            return
    candidates.append(axis)

def _generate_local_axis_refinement_candidates(
    base_axis: np.ndarray,
    angle_degrees: Tuple[float, ...] = (2.0, 4.0, 6.0, 8.0, 12.0),
    azimuth_steps: int = 12,
) -> list[np.ndarray]:
    base_axis = normalize(base_axis)
    frame = orthonormal_basis_from_z(base_axis)
    tangent_x = frame[:, 0]
    tangent_y = frame[:, 1]

    candidates: list[np.ndarray] = []
    _append_unique_axis(candidates, base_axis)
    _append_unique_axis(candidates, -base_axis)
    for angle_deg in angle_degrees:
        tangent_scale = np.tan(np.deg2rad(angle_deg))
        for k in range(azimuth_steps):
            phi = 2.0 * np.pi * k / azimuth_steps
            tangent = np.cos(phi) * tangent_x + np.sin(phi) * tangent_y
            _append_unique_axis(candidates, normalize(base_axis + tangent_scale * tangent))
            _append_unique_axis(candidates, normalize(-base_axis + tangent_scale * tangent))
    return candidates

def _generate_pca_axis_candidates(points: np.ndarray) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    _, eigvecs, _ = pca_axes(points)
    for i in range(3):
        axis = normalize(eigvecs[:, i])
        _append_unique_axis(candidates, axis)
        _append_unique_axis(candidates, -axis)
    return candidates

def _generate_frustum_axis_candidates(
    points: np.ndarray,
    cylinder_axis_hint: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    if cylinder_axis_hint is not None:
        for axis in _generate_local_axis_refinement_candidates(cylinder_axis_hint):
            _append_unique_axis(candidates, axis)

    for axis in _generate_pca_axis_candidates(points):
        _append_unique_axis(candidates, axis)
    return candidates

def _generate_cuboid_rotation_candidates(eigvecs: np.ndarray) -> list[np.ndarray]:
    eigvecs = np.asarray(eigvecs, dtype=np.float64)
    candidates: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        base_x = normalize(eigvecs[:, perm[0]])
        base_y = normalize(eigvecs[:, perm[1]])
        for sign_x in (+1.0, -1.0):
            for sign_y in (+1.0, -1.0):
                x_axis = sign_x * base_x
                y_axis = sign_y * base_y
                z_axis = normalize(np.cross(x_axis, y_axis))
                R = make_right_handed(np.column_stack([x_axis, y_axis, z_axis]))
                candidates.append(R)
    return candidates

def _select_frustum_sidewall_slices(
    a_arr: np.ndarray,
    b_arr: np.ndarray,
    w_arr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    major_arr = np.maximum(np.asarray(a_arr, dtype=np.float64), np.asarray(b_arr, dtype=np.float64))
    minor_arr = np.minimum(np.asarray(a_arr, dtype=np.float64), np.asarray(b_arr, dtype=np.float64))
    weights = np.asarray(w_arr, dtype=np.float64)
    slice_aspect = np.clip(major_arr / np.maximum(minor_arr, 1e-8), 1.0, 20.0)
    reference_aspect = float(np.exp(np.average(np.log(slice_aspect), weights=weights)))
    aspect_threshold = max(1.8, 0.72 * reference_aspect)
    sidewall_mask = slice_aspect >= aspect_threshold
    if int(sidewall_mask.sum()) < 3:
        sidewall_mask = np.ones_like(slice_aspect, dtype=bool)
    return major_arr, minor_arr, slice_aspect, sidewall_mask, reference_aspect

def _fit_frustum_given_axis(
    points: np.ndarray,
    axis: np.ndarray,
    robust_percentile: Optional[float] = 1.0,
    shape_hint: Optional[Tuple[np.ndarray, float, float]] = None,
    allow_axis_flip: bool = True,
) -> FrustumFitResult:
    """给定轴向，拟合“共享 taper 的椭圆截锥侧面”模型。"""
    points = ensure_points(points)
    axis = normalize(axis)

    centroid = points.mean(axis=0)
    R_frame = orthonormal_basis_from_z(axis)
    local = transform_world_to_local(points, centroid, R_frame)
    xy = local[:, :2]
    w = local[:, 2]

    ecx, ecy, _, _, ellipse_angle = fit_ellipse_2d(xy)
    hint_aspect: Optional[float] = None
    major_dir_world_hint: Optional[np.ndarray] = None
    if shape_hint is not None:
        major_dir_world_hint = normalize(np.asarray(shape_hint[0], dtype=np.float64))
        projected_hint = major_dir_world_hint - np.dot(major_dir_world_hint, axis) * axis
        if np.linalg.norm(projected_hint) > 1e-6:
            projected_hint = normalize(projected_hint)
            local_hint = projected_hint @ R_frame
            ellipse_angle = float(np.arctan2(local_hint[1], local_hint[0]))
            hint_aspect = float(max(shape_hint[1] / max(shape_hint[2], 1e-8), 1.0))
        else:
            major_dir_world_hint = None
    ca, sa = np.cos(ellipse_angle), np.sin(ellipse_angle)
    uv_basis = np.array([
        [ca, -sa],
        [sa,  ca],
    ], dtype=np.float64)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        uv = (xy - np.array([ecx, ecy], dtype=np.float64)) @ uv_basis

    p = 1.0 if robust_percentile is None else float(robust_percentile)
    p = min(max(p, 0.1), 10.0)
    z_min = float(np.percentile(w, p))
    z_max = float(np.percentile(w, 100.0 - p))
    height = float(z_max - z_min)
    if height <= 1e-6:
        raise ValueError("Estimated frustum height is too small.")

    axial_ratio = np.clip((w - z_min) / max(height, 1e-8), 0.0, 1.0)
    slice_centers = np.linspace(0.05, 0.95, 10)
    slice_half_width = 0.08
    min_slice_points = max(50, int(0.02 * len(points)))

    ts: list[float] = []
    a_vals: list[float] = []
    b_vals: list[float] = []
    weights: list[float] = []

    for t_center in slice_centers:
        mask = np.abs(axial_ratio - t_center) <= slice_half_width
        if int(mask.sum()) < min_slice_points:
            continue

        u_slice = uv[mask, 0]
        v_slice = uv[mask, 1]
        a_slice = 0.5 * float(np.percentile(u_slice, 100.0 - p) - np.percentile(u_slice, p))
        b_slice = 0.5 * float(np.percentile(v_slice, 100.0 - p) - np.percentile(v_slice, p))
        if min(a_slice, b_slice) <= 1e-5:
            continue

        ts.append(float(t_center))
        a_vals.append(a_slice)
        b_vals.append(b_slice)
        weights.append(float(mask.sum()))

    if len(ts) < 3:
        raise ValueError("Not enough valid frustum slices.")

    ts_arr = np.asarray(ts, dtype=np.float64)
    a_arr = np.asarray(a_vals, dtype=np.float64)
    b_arr = np.asarray(b_vals, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)

    major_arr, minor_arr, slice_aspect, sidewall_mask, reference_aspect = _select_frustum_sidewall_slices(
        a_arr, b_arr, w_arr
    )
    ts_fit = ts_arr[sidewall_mask]
    major_fit = major_arr[sidewall_mask]
    minor_fit = minor_arr[sidewall_mask]
    w_fit = w_arr[sidewall_mask]

    # End caps become nearly circular and can dominate the last few slices.
    # Use side-wall-dominant slices to infer taper direction.
    major_coef = np.polyfit(ts_fit, major_fit, 1, w=w_fit)
    if allow_axis_flip and float(np.polyval(major_coef, 1.0)) < float(np.polyval(major_coef, 0.0)) - 1e-6:
        return _fit_frustum_given_axis(
            points=points,
            axis=-axis,
            robust_percentile=robust_percentile,
            shape_hint=shape_hint,
            allow_axis_flip=False,
        )

    observed_aspect = float(np.exp(np.average(np.log(slice_aspect[sidewall_mask]), weights=w_fit)))
    if hint_aspect is None:
        hint_aspect = observed_aspect
    else:
        hint_aspect = float(np.sqrt(max(hint_aspect * observed_aspect, 1.0)))

    major_unit = float(np.sqrt(max(hint_aspect, 1.0)))
    minor_unit = float(1.0 / max(major_unit, 1e-8))
    denom = major_unit * major_unit + minor_unit * minor_unit
    scale_obs = (a_arr * major_unit + b_arr * minor_unit) / max(denom, 1e-8)
    scale_weights = w_arr.copy()
    scale_weights[~sidewall_mask] *= 0.35
    scale_coef = np.polyfit(ts_arr, scale_obs, 1, w=scale_weights)
    small_scale = float(np.polyval(scale_coef, 0.0))
    large_scale = float(np.polyval(scale_coef, 1.0))

    scale_lo, scale_hi = np.percentile(scale_obs, [5.0, 95.0])
    small_scale = float(np.clip(small_scale, 0.8 * scale_lo, 1.2 * scale_hi))
    large_scale = float(np.clip(large_scale, 0.8 * scale_lo, 1.2 * scale_hi))
    small_end_major = float(small_scale * major_unit)
    small_end_minor = float(small_scale * minor_unit)
    large_end_major = float(large_scale * major_unit)
    large_end_minor = float(large_scale * minor_unit)
    if min(small_end_major, small_end_minor, large_end_major, large_end_minor, small_scale, large_scale) <= 1e-6:
        raise ValueError("Invalid frustum radii from fit.")

    max_aspect = max(
        small_end_major / max(small_end_minor, 1e-8),
        large_end_major / max(large_end_minor, 1e-8),
    )
    if max(max_aspect, reference_aspect) <= 1.12:
        major_dir_world = _stable_inplane_major_direction(
            axis,
            reference_dirs=(R_frame[:, 0], R_frame[:, 1]),
        )
    elif major_dir_world_hint is not None:
        major_dir_world = normalize(major_dir_world_hint - np.dot(major_dir_world_hint, axis) * axis)
    else:
        major_dir_world = R_frame @ np.array([ca, sa, 0.0], dtype=np.float64)
        major_dir_world = normalize(major_dir_world)
    rotation = _make_frame_from_major_axis(major_dir_world, axis)

    small_end_center_local = np.array([ecx, ecy, z_min], dtype=np.float64)
    large_end_center_local = np.array([ecx, ecy, z_max], dtype=np.float64)
    center_local = np.array([ecx, ecy, 0.5 * (z_min + z_max)], dtype=np.float64)
    small_end_center_world = transform_local_to_world(small_end_center_local[None], centroid, R_frame)[0]
    large_end_center_world = transform_local_to_world(large_end_center_local[None], centroid, R_frame)[0]
    center_world = transform_local_to_world(center_local[None], centroid, R_frame)[0]

    residual = float(np.mean(point_to_ellip_frustum_distance(
        points=points,
        small_end_center=small_end_center_world,
        rotation=rotation,
        small_end_major=small_end_major,
        small_end_minor=small_end_minor,
        large_end_major=large_end_major,
        large_end_minor=large_end_minor,
        height=height,
    )))

    return FrustumFitResult(
        center=center_world,
        small_end_center=small_end_center_world,
        large_end_center=large_end_center_world,
        axis=axis,
        small_end_major=small_end_major,
        small_end_minor=small_end_minor,
        large_end_major=large_end_major,
        large_end_minor=large_end_minor,
        height=height,
        rotation=rotation,
        pose_type=None,
        residual=residual,
    )

def _is_reasonable_frustum_fit(frustum: FrustumFitResult, points: np.ndarray) -> bool:
    spans = np.ptp(points, axis=0)
    object_scale = float(max(np.linalg.norm(spans), 1e-6))
    max_span = float(max(np.max(spans), 1e-6))
    small_scale = float(np.sqrt(max(frustum.small_end_major * frustum.small_end_minor, 1e-12)))
    large_scale = float(np.sqrt(max(frustum.large_end_major * frustum.large_end_minor, 1e-12)))

    if not np.all(np.isfinite([
        frustum.small_end_major, frustum.small_end_minor,
        frustum.large_end_major, frustum.large_end_minor,
        frustum.height, frustum.residual,
    ])):
        return False
    if frustum.small_end_major < frustum.small_end_minor or frustum.large_end_major < frustum.large_end_minor:
        return False
    if large_scale <= small_scale:
        return False
    if frustum.height <= max(0.015, 0.08 * object_scale):
        return False
    if frustum.small_end_minor <= max(0.003, 0.012 * object_scale):
        return False
    if frustum.large_end_minor <= max(0.005, 0.02 * object_scale):
        return False
    if frustum.large_end_major > max(0.06, 2.0 * max_span):
        return False
    if frustum.height > max(0.18, 2.5 * object_scale):
        return False

    taper_ratio = large_scale / max(small_scale, 1e-8)
    if taper_ratio < 1.05:
        return False
    if taper_ratio > 2.5:
        return False

    if frustum.height / max(large_scale, 1e-8) < 1.8:
        return False
    return True

def _frustum_equivalent_scales(frustum: FrustumFitResult) -> Tuple[float, float]:
    small_scale = float(np.sqrt(max(frustum.small_end_major * frustum.small_end_minor, 1e-12)))
    large_scale = float(np.sqrt(max(frustum.large_end_major * frustum.large_end_minor, 1e-12)))
    return small_scale, large_scale

def _evaluate_frustum_cylinder_preference(
    frustum: FrustumFitResult,
    cylinder: CylinderFitResult,
) -> Dict[str, Any]:
    small_scale, large_scale = _frustum_equivalent_scales(frustum)
    taper_ratio = float(large_scale / max(small_scale, 1e-8))
    taper_delta = float(large_scale - small_scale)
    residual_gap = float(frustum.residual - cylinder.residual)
    relative_gap = float(residual_gap / max(cylinder.residual, 1e-8))
    axis_alignment = float(abs(np.dot(normalize(frustum.axis), normalize(cylinder.axis))))
    height_ratio = float(frustum.height / max(cylinder.height, 1e-8))

    prefer_frustum = bool(
        residual_gap <= 1.0e-3 and
        relative_gap <= 1.0 and
        taper_ratio >= 1.10 and
        taper_delta >= max(0.0020, 0.08 * large_scale) and
        axis_alignment >= np.cos(np.deg2rad(20.0)) and
        height_ratio >= 0.85
    )

    return {
        "prefer_frustum": prefer_frustum,
        "residual_gap": residual_gap,
        "relative_gap": relative_gap,
        "taper_ratio": taper_ratio,
        "taper_delta": taper_delta,
        "axis_alignment": axis_alignment,
        "height_ratio": height_ratio,
        "residual_gap_threshold": 1.0e-3,
        "relative_gap_threshold": 1.0,
        "taper_ratio_threshold": 1.10,
        "taper_delta_threshold": float(max(0.0020, 0.08 * large_scale)),
        "axis_alignment_threshold": float(np.cos(np.deg2rad(20.0))),
        "height_ratio_threshold": 0.85,
    }

def _fit_bowl_given_axis(
    points: np.ndarray,
    axis: np.ndarray,
    robust_percentile: Optional[float] = 1.0,
) -> BowlFitResult:
    """给定碗轴方向，拟合椭圆抛物面开口曲面。"""
    points = ensure_points(points)
    axis = normalize(axis)
    centroid = points.mean(axis=0)
    R_frame = orthonormal_basis_from_z(axis)
    local = transform_world_to_local(points, centroid, R_frame)

    u = local[:, 0]
    v = local[:, 1]
    w = local[:, 2]

    A = np.column_stack([u**2, u * v, v**2, u, v, np.ones_like(u)])
    coeff, *_ = np.linalg.lstsq(A, w, rcond=None)
    a, b, c, d, e, f0 = coeff

    Q = np.array([[a, 0.5 * b], [0.5 * b, c]], dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(Q)
    if not np.all(np.isfinite(eigvals)):
        raise ValueError("Bowl quadratic fit produced non-finite curvature.")
    if eigvals[0] <= 1e-6:
        raise ValueError("Quadratic form is not positive definite; not a bowl.")

    uv_vertex = -0.5 * np.linalg.solve(Q, np.array([d, e], dtype=np.float64))
    u0, v0 = float(uv_vertex[0]), float(uv_vertex[1])
    w0 = float(a * u0 * u0 + b * u0 * v0 + c * v0 * v0 + d * u0 + e * v0 + f0)

    uv_centered = np.column_stack([u - u0, v - v0])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        uv_principal = uv_centered @ eigvecs
    major_coord = uv_principal[:, 0]
    minor_coord = uv_principal[:, 1]
    w_rel = w - w0
    w_pred_rel = eigvals[0] * major_coord**2 + eigvals[1] * minor_coord**2

    non_negative_w = np.maximum(w_rel, 0.0)
    if robust_percentile is None:
        depth = float(non_negative_w.max())
    else:
        p = float(robust_percentile)
        depth = float(np.percentile(non_negative_w, 100.0 - p))
    if depth <= 1e-6:
        raise ValueError("Estimated bowl depth is too small.")

    rim_major = float(np.sqrt(depth / eigvals[0]))
    rim_minor = float(np.sqrt(depth / eigvals[1]))
    if not np.isfinite(rim_major) or not np.isfinite(rim_minor):
        raise ValueError("Invalid rim size from bowl fit.")

    major_dir_world = R_frame @ np.array([eigvecs[0, 0], eigvecs[1, 0], 0.0], dtype=np.float64)
    rotation = _make_frame_from_major_axis(major_dir_world, axis)

    vertex_local = np.array([u0, v0, w0], dtype=np.float64)
    rim_center_local = np.array([u0, v0, w0 + depth], dtype=np.float64)
    vertex_world = transform_local_to_world(vertex_local[None], centroid, R_frame)[0]
    rim_center_world = transform_local_to_world(rim_center_local[None], centroid, R_frame)[0]

    grad_sq = (2.0 * eigvals[0] * major_coord) ** 2 + (2.0 * eigvals[1] * minor_coord) ** 2
    residuals = np.abs(w_rel - w_pred_rel) / np.sqrt(1.0 + grad_sq)
    residual = float(np.mean(residuals))

    return BowlFitResult(
        vertex=vertex_world,
        rim_center=rim_center_world,
        axis=axis,
        rotation=rotation,
        rim_major=rim_major,
        rim_minor=rim_minor,
        depth=depth,
        pose_type=None,
        residual=residual,
    )

def _is_reasonable_bowl_fit(bowl: BowlFitResult, points: np.ndarray) -> bool:
    spans = np.ptp(points, axis=0)
    object_scale = float(max(np.linalg.norm(spans), 1e-6))
    max_span = float(max(np.max(spans), 1e-6))

    if not np.all(np.isfinite([
        bowl.rim_major, bowl.rim_minor, bowl.depth, bowl.residual,
    ])):
        return False
    if bowl.rim_major < bowl.rim_minor:
        return False
    if bowl.depth <= max(0.005, 0.02 * object_scale):
        return False
    if bowl.rim_minor <= max(0.005, 0.02 * object_scale):
        return False
    if bowl.rim_major > max(0.05, 2.0 * max_span):
        return False
    if bowl.depth > max(0.05, 2.0 * object_scale):
        return False
    if bowl.rim_major / max(bowl.rim_minor, 1e-8) > 8.0:
        return False
    depth_ratio = _dish_depth_ratio(bowl.depth, bowl.rim_minor)
    if depth_ratio < 0.28 or depth_ratio > 1.25:
        return False

    min_curvature = bowl.depth / max(bowl.rim_major**2, 1e-12)
    if min_curvature <= 1e-3:
        return False
    return True

def _smoothstep01(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)

def _fit_plate_given_axis(
    points: np.ndarray,
    axis: np.ndarray,
    robust_percentile: Optional[float] = 1.0,
) -> PlateFitResult:
    """给定盘轴方向，拟合“平底 + 上翘盘沿”的浅盘模型。"""
    points = ensure_points(points)
    axis = normalize(axis)

    centroid = points.mean(axis=0)
    R_frame = orthonormal_basis_from_z(axis)
    local = transform_world_to_local(points, centroid, R_frame)
    xy = local[:, :2]
    w = local[:, 2]

    # 盘子顶视图更接近“填充椭圆”而不是边界点，因此不用边界椭圆拟合，
    # 而是用 2D PCA + 分位数盒来稳健估计中心、方向和盘沿尺度。
    xy_mean = xy.mean(axis=0)
    xy_centered = xy - xy_mean
    cov2 = np.cov(xy_centered.T)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    order = np.argsort(eigvals2)[::-1]
    eigvecs2 = eigvecs2[:, order]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        uv0 = xy_centered @ eigvecs2

    p = 1.0 if robust_percentile is None else float(robust_percentile)
    p = min(max(p, 0.1), 10.0)
    u_min, u_max = np.percentile(uv0[:, 0], [p, 100.0 - p])
    v_min, v_max = np.percentile(uv0[:, 1], [p, 100.0 - p])
    u0 = 0.5 * (u_min + u_max)
    v0 = 0.5 * (v_min + v_max)
    rim_major = 0.5 * (u_max - u_min)
    rim_minor = 0.5 * (v_max - v_min)
    if rim_major <= 1e-6 or rim_minor <= 1e-6:
        raise ValueError("Estimated plate rim size is too small.")
    if rim_minor > rim_major:
        rim_major, rim_minor = rim_minor, rim_major
        eigvecs2 = eigvecs2[:, [1, 0]]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            uv0 = xy_centered @ eigvecs2
        u_min, u_max = np.percentile(uv0[:, 0], [p, 100.0 - p])
        v_min, v_max = np.percentile(uv0[:, 1], [p, 100.0 - p])
        u0 = 0.5 * (u_min + u_max)
        v0 = 0.5 * (v_min + v_max)

    center_xy = xy_mean + eigvecs2 @ np.array([u0, v0], dtype=np.float64)
    major_coord = uv0[:, 0] - u0
    minor_coord = uv0[:, 1] - v0
    rho = np.sqrt((major_coord / rim_major) ** 2 + (minor_coord / rim_minor) ** 2)

    candidate_ratios = np.linspace(0.35, 0.9, 12)
    best = None
    best_score = np.inf
    scale_xy = min(rim_major, rim_minor)

    for flat_ratio in candidate_ratios:
        inner_mask = rho <= flat_ratio
        outer_mask = rho >= min(flat_ratio + 0.08, 0.95)
        if inner_mask.sum() < max(30, int(0.08 * len(points))):
            continue
        if outer_mask.sum() < max(20, int(0.05 * len(points))):
            continue

        w0 = float(np.median(w[inner_mask]))
        transition = np.clip((rho - flat_ratio) / max(1.0 - flat_ratio, 1e-8), 0.0, 1.0)
        profile = _smoothstep01(transition)
        denom = float(np.dot(profile, profile))
        if denom < 1e-10:
            continue

        depth = float(np.dot(profile, (w - w0)) / denom)
        if depth <= 1e-6:
            continue

        pred = w0 + depth * profile
        radial_excess = np.maximum(rho - 1.0, 0.0) * scale_xy
        vertical_err = w - pred
        dist = np.sqrt(vertical_err**2 + radial_excess**2)
        residual = float(np.mean(dist))
        bottom_flatness = float(np.std(w[inner_mask] - w0))
        outer_target = float(np.percentile(w[outer_mask], 75.0) - w0)
        rim_consistency = abs(outer_target - depth)
        score = residual + 0.3 * bottom_flatness + 0.15 * rim_consistency

        if score < best_score:
            best_score = score
            best = (flat_ratio, w0, depth, residual, bottom_flatness)

    if best is None:
        raise ValueError("All plate flat-bottom candidates failed.")

    flat_ratio, w0, depth, residual, bottom_flatness = best
    bottom_major = float(flat_ratio * rim_major)
    bottom_minor = float(flat_ratio * rim_minor)

    major_dir_world = R_frame @ np.array([eigvecs2[0, 0], eigvecs2[1, 0], 0.0], dtype=np.float64)
    rotation = _make_frame_from_major_axis(major_dir_world, axis)

    vertex_local = np.array([center_xy[0], center_xy[1], w0], dtype=np.float64)
    rim_center_local = np.array([center_xy[0], center_xy[1], w0 + depth], dtype=np.float64)
    vertex_world = transform_local_to_world(vertex_local[None], centroid, R_frame)[0]
    rim_center_world = transform_local_to_world(rim_center_local[None], centroid, R_frame)[0]

    return PlateFitResult(
        vertex=vertex_world,
        rim_center=rim_center_world,
        axis=axis,
        rotation=rotation,
        bottom_major=bottom_major,
        bottom_minor=bottom_minor,
        rim_major=float(rim_major),
        rim_minor=float(rim_minor),
        depth=float(depth),
        pose_type=None,
        bottom_flatness=bottom_flatness,
        residual=float(residual),
    )

def _is_reasonable_plate_fit(plate: PlateFitResult, points: np.ndarray) -> bool:
    spans = np.ptp(points, axis=0)
    object_scale = float(max(np.linalg.norm(spans), 1e-6))
    max_span = float(max(np.max(spans), 1e-6))

    if not np.all(np.isfinite([
        plate.bottom_major, plate.bottom_minor, plate.rim_major, plate.rim_minor,
        plate.depth, plate.bottom_flatness, plate.residual,
    ])):
        return False
    if plate.rim_major < plate.rim_minor:
        return False
    if plate.bottom_major < plate.bottom_minor:
        return False
    if plate.bottom_major >= plate.rim_major or plate.bottom_minor >= plate.rim_minor:
        return False
    if plate.depth <= max(0.001, 0.005 * object_scale):
        return False
    if plate.rim_minor <= max(0.005, 0.02 * object_scale):
        return False
    if plate.rim_major > max(0.05, 2.0 * max_span):
        return False
    if plate.depth > max(0.03, 0.45 * object_scale):
        return False
    if plate.rim_major / max(plate.rim_minor, 1e-8) > 8.0:
        return False

    flat_ratio = plate.bottom_minor / max(plate.rim_minor, 1e-8)
    if flat_ratio < 0.38 or flat_ratio > 0.96:
        return False

    depth_ratio = _dish_depth_ratio(plate.depth, plate.rim_minor)
    if depth_ratio < 0.01 or depth_ratio > 0.22:
        return False

    if plate.bottom_flatness > max(0.004, 0.5 * plate.depth):
        return False
    return True

def fit_cuboid(points: np.ndarray, robust_percentile: Optional[float] = 1.0) -> CuboidFitResult:
    # 不依赖竖直参考，直接枚举 PCA 主轴的排列和符号得到候选 OBB。
    points = ensure_points(points)
    centroid, eigvecs, _ = pca_axes(points)
    best = None
    best_residual = np.inf

    for R in _generate_cuboid_rotation_candidates(eigvecs):
        local = transform_world_to_local(points, centroid, R)
        if robust_percentile is None:
            min_local = local.min(axis=0)
            max_local = local.max(axis=0)
        else:
            p = float(robust_percentile)
            min_local = np.percentile(local, p, axis=0)
            max_local = np.percentile(local, 100.0 - p, axis=0)

        residual = float(np.mean(point_to_aabb_surface_distance(local, min_local, max_local)))
        if residual < best_residual:
            best_residual = residual
            best = (R, local, min_local, max_local)

    R, local, min_local, max_local = best
    size = max_local - min_local
    center_local = 0.5 * (min_local + max_local)
    center_world = transform_local_to_world(center_local[None], centroid, R)[0]
    corners_world = transform_local_to_world(cuboid_corners_from_bounds(min_local, max_local), centroid, R)

    face_poses = cuboid_face_poses_wxyz(center_world, R, size)

    return CuboidFitResult(
        center=center_world,
        rotation=R,
        size=size,
        min_local=min_local,
        max_local=max_local,
        corners_world=corners_world,
        face_poses_wxyz=face_poses,
        residual=best_residual,
    )

def _fit_cylinder_given_axis(
    points: np.ndarray, axis: np.ndarray, robust_percentile: float = 1.0,
) -> CylinderFitResult:
    """给定柱轴方向，拟合椭圆柱截面和高度。"""
    points = ensure_points(points)
    axis = normalize(axis)
    centroid = points.mean(axis=0)
    R_frame = orthonormal_basis_from_z(axis)
    local = transform_world_to_local(points, centroid, R_frame)
    xy, z_vals = local[:, :2], local[:, 2]

    ecx, ecy, semi_major, semi_minor, ellipse_angle = fit_ellipse_2d(xy)

    p = float(robust_percentile)
    z_min = float(np.percentile(z_vals, p))
    z_max = float(np.percentile(z_vals, 100.0 - p))
    height = z_max - z_min
    z_center = 0.5 * (z_min + z_max)

    center_local = np.array([ecx, ecy, z_center])
    center_world = transform_local_to_world(center_local[None], centroid, R_frame)[0]
    top_world = transform_local_to_world(np.array([[ecx, ecy, z_max]]), centroid, R_frame)[0]
    bottom_world = transform_local_to_world(np.array([[ecx, ecy, z_min]]), centroid, R_frame)[0]

    ca, sa = np.cos(ellipse_angle), np.sin(ellipse_angle)
    major_dir = R_frame @ np.array([ca, sa, 0.0])
    minor_dir = R_frame @ np.array([-sa, ca, 0.0])
    if _is_nearly_round_cross_section(semi_major, semi_minor):
        rotation = _make_frame_from_major_axis(
            _stable_inplane_major_direction(axis, reference_dirs=(R_frame[:, 0], R_frame[:, 1])),
            axis,
        )
    else:
        rotation = make_right_handed(np.column_stack([major_dir, minor_dir, axis]))

    residual = float(np.mean(point_to_ellip_cylinder_distance(
        points, center_world, rotation, semi_major, semi_minor, height,
    )))

    return CylinderFitResult(
        center=center_world, axis=axis,
        semi_major=semi_major, semi_minor=semi_minor,
        height=height, top_center=top_world, bottom_center=bottom_world,
        rotation=rotation, residual=residual,
    )


def fit_cylinder(
    points: np.ndarray,
    robust_percentile: float = 1.0,
) -> CylinderFitResult:
    """尝试 PCA 轴向候选，选残差最小的椭圆柱拟合。"""
    points = ensure_points(points)
    candidates = _generate_pca_axis_candidates(points)

    best: Optional[CylinderFitResult] = None
    for axis_cand in candidates:
        try:
            res = _fit_cylinder_given_axis(points, axis_cand, robust_percentile)
        except Exception:
            continue
        if best is None or res.residual < best.residual:
            best = res

    if best is None:
        raise ValueError("所有轴向候选均拟合失败。")
    return best

def fit_frustum(
    points: np.ndarray,
    robust_percentile: Optional[float] = 1.0,
    cylinder_hint: Optional[CylinderFitResult] = None,
) -> FrustumFitResult:
    """尝试多种轴向候选，选残差最小的椭圆截锥拟合。"""
    points = ensure_points(points)
    if cylinder_hint is None:
        try:
            cylinder_hint = fit_cylinder(points, robust_percentile=robust_percentile)
        except Exception:
            cylinder_hint = None

    cylinder_axis_hint = None if cylinder_hint is None else cylinder_hint.axis
    candidates = _generate_frustum_axis_candidates(points, cylinder_axis_hint=cylinder_axis_hint)

    best_reasonable: Optional[FrustumFitResult] = None
    best_reasonable_taper = -np.inf
    best_any: Optional[FrustumFitResult] = None
    best_any_taper = -np.inf
    tie_eps = 1e-4

    def _evaluate_axis(axis_cand: np.ndarray) -> None:
        nonlocal best_any, best_any_taper, best_reasonable, best_reasonable_taper
        shape_hints: list[Optional[Tuple[np.ndarray, float, float]]] = [None]
        if cylinder_hint is not None and abs(float(np.dot(normalize(axis_cand), normalize(cylinder_hint.axis)))) >= np.cos(np.deg2rad(25.0)):
            shape_hints = [
                (cylinder_hint.rotation[:, 0], cylinder_hint.semi_major, cylinder_hint.semi_minor),
                (cylinder_hint.rotation[:, 1], cylinder_hint.semi_major, cylinder_hint.semi_minor),
                None,
            ]

        for shape_hint in shape_hints:
            try:
                res = _fit_frustum_given_axis(
                    points=points,
                    axis=axis_cand,
                    robust_percentile=robust_percentile,
                    shape_hint=shape_hint,
                )
            except Exception:
                continue

            taper_ratio = np.sqrt(
                (res.large_end_major * res.large_end_minor) /
                max(res.small_end_major * res.small_end_minor, 1e-12)
            )
            if (
                best_any is None or
                res.residual < best_any.residual - tie_eps or
                (abs(res.residual - best_any.residual) <= tie_eps and taper_ratio > best_any_taper)
            ):
                best_any = res
                best_any_taper = taper_ratio

            if not _is_reasonable_frustum_fit(res, points):
                continue
            if (
                best_reasonable is None or
                res.residual < best_reasonable.residual - tie_eps or
                (abs(res.residual - best_reasonable.residual) <= tie_eps and taper_ratio > best_reasonable_taper)
            ):
                best_reasonable = res
                best_reasonable_taper = taper_ratio

    for axis_cand in candidates:
        _evaluate_axis(axis_cand)

    refinement_seeds: list[np.ndarray] = []
    if best_any is not None:
        refinement_seeds.append(best_any.axis)
    if best_reasonable is not None and best_any is not None:
        if abs(float(np.dot(normalize(best_reasonable.axis), normalize(best_any.axis)))) < 0.999:
            refinement_seeds.append(best_reasonable.axis)

    for seed_axis in refinement_seeds:
        refined_candidates = _generate_local_axis_refinement_candidates(
            seed_axis,
            angle_degrees=(1.0, 2.0, 3.0),
            azimuth_steps=24,
        )
        for axis_cand in refined_candidates:
            _evaluate_axis(axis_cand)

    best = best_reasonable if best_reasonable is not None else best_any
    if best is None:
        raise ValueError("所有 frustum 轴向候选均拟合失败。")
    return best

def fit_bowl(
    points: np.ndarray,
    robust_percentile: Optional[float] = 1.0,
) -> BowlFitResult:
    """通过 PCA 轴向候选搜索最佳 bowl 拟合。"""
    points = ensure_points(points)
    candidates = _generate_pca_axis_candidates(points)

    best: Optional[BowlFitResult] = None
    for axis_cand in candidates:
        try:
            res = _fit_bowl_given_axis(
                points=points,
                axis=axis_cand,
                robust_percentile=robust_percentile,
            )
        except Exception:
            continue
        if best is None or res.residual < best.residual:
            best = res

    if best is None:
        raise ValueError("所有 bowl 轴向候选均拟合失败。")
    return best

def fit_plate(
    points: np.ndarray,
    robust_percentile: Optional[float] = 1.0,
) -> PlateFitResult:
    points = ensure_points(points)
    candidates = _generate_pca_axis_candidates(points)

    best: Optional[PlateFitResult] = None
    for axis_cand in candidates:
        try:
            res = _fit_plate_given_axis(
                points=points,
                axis=axis_cand,
                robust_percentile=robust_percentile,
            )
        except Exception:
            continue
        if best is None or res.residual < best.residual:
            best = res

    if best is None:
        raise ValueError("所有 plate 轴向候选均拟合失败。")
    return best


def fit_best_primitive(
    points: np.ndarray,
    robust_percentile: float = 1.0,
    fit_type: Optional[str] = None,
) -> Dict[str, Any]:
    fit_search_start = time.perf_counter()
    points = ensure_points(points)
    valid_fit_types = {None, "cuboid", "cylinder", "frustum", "bowl", "plate"}
    if fit_type not in valid_fit_types:
        raise ValueError(f"fit_type must be one of {sorted(t for t in valid_fit_types if t is not None)}, got {fit_type}")

    scores: Dict[str, float] = {}
    raw_scores: Dict[str, float] = {}
    all_results: Dict[str, Any] = {}
    filtered_out: Dict[str, Dict[str, Any]] = {}
    fit_type_timings: Dict[str, float] = {}
    cylinder_result: Optional[CylinderFitResult] = None

    def _record_fit_success(
        name: str,
        result: Any,
        reasonableness_fn: Optional[Any] = None,
    ) -> None:
        all_results[name] = result
        raw_scores[name] = float(result.residual)
        if fit_type == name or reasonableness_fn is None:
            scores[name] = float(result.residual)
            return
        if reasonableness_fn(result, points):
            scores[name] = float(result.residual)
            return
        filtered_out[name] = {
            "status": "not_reasonable",
            "reason": "not reasonable for current points",
            "raw_residual": float(result.residual),
        }

    def _record_fit_failure(name: str, exc: Exception) -> None:
        filtered_out[name] = {
            "status": "fit_failed",
            "reason": f"{type(exc).__name__}: {exc}",
        }

    if fit_type in (None, "cuboid"):
        fit_start = time.perf_counter()
        try:
            cuboid = fit_cuboid(points, robust_percentile=robust_percentile)
        except Exception as exc:
            if fit_type == "cuboid":
                raise
            _record_fit_failure("cuboid", exc)
        else:
            _record_fit_success("cuboid", cuboid)
        finally:
            fit_type_timings["cuboid"] = time.perf_counter() - fit_start

    if fit_type in (None, "cylinder"):
        fit_start = time.perf_counter()
        try:
            cylinder = fit_cylinder(points, robust_percentile=robust_percentile)
        except Exception as exc:
            if fit_type == "cylinder":
                raise
            _record_fit_failure("cylinder", exc)
        else:
            _record_fit_success("cylinder", cylinder)
            cylinder_result = cylinder
        finally:
            fit_type_timings["cylinder"] = time.perf_counter() - fit_start

    if fit_type in (None, "frustum"):
        fit_start = time.perf_counter()
        try:
            frustum = fit_frustum(
                points,
                robust_percentile=robust_percentile,
                cylinder_hint=cylinder_result,
            )
        except Exception as exc:
            if fit_type == "frustum":
                raise
            _record_fit_failure("frustum", exc)
        else:
            _record_fit_success("frustum", frustum, reasonableness_fn=_is_reasonable_frustum_fit)
            if fit_type == "frustum" and "frustum" not in scores:
                raise ValueError("frustum fit is not reasonable for current points.")
        finally:
            fit_type_timings["frustum"] = time.perf_counter() - fit_start

    if fit_type in (None, "bowl"):
        fit_start = time.perf_counter()
        try:
            bowl = fit_bowl(points, robust_percentile=robust_percentile)
        except Exception as exc:
            if fit_type == "bowl":
                raise
            _record_fit_failure("bowl", exc)
        else:
            _record_fit_success("bowl", bowl, reasonableness_fn=_is_reasonable_bowl_fit)
            if fit_type == "bowl" and "bowl" not in scores:
                raise ValueError("bowl fit is not reasonable for current points.")
        finally:
            fit_type_timings["bowl"] = time.perf_counter() - fit_start

    if fit_type in (None, "plate"):
        fit_start = time.perf_counter()
        try:
            plate = fit_plate(points, robust_percentile=robust_percentile)
        except Exception as exc:
            if fit_type == "plate":
                raise
            _record_fit_failure("plate", exc)
        else:
            _record_fit_success("plate", plate, reasonableness_fn=_is_reasonable_plate_fit)
            if fit_type == "plate" and "plate" not in scores:
                raise ValueError("plate fit is not reasonable for current points.")
        finally:
            fit_type_timings["plate"] = time.perf_counter() - fit_start

    if not scores:
        raise ValueError("No valid primitive fit result.")
    best_type = min(scores, key=scores.get)
    selection_info: Optional[Dict[str, Any]] = None
    if (
        fit_type is None and
        best_type == "cylinder" and
        "frustum" in scores and
        "cylinder" in scores
    ):
        frustum_vs_cylinder = _evaluate_frustum_cylinder_preference(
            frustum=all_results["frustum"],
            cylinder=all_results["cylinder"],
        )
        selection_info = {
            "strategy": "min_residual",
            "frustum_cylinder_resolution": frustum_vs_cylinder,
        }
        if frustum_vs_cylinder["prefer_frustum"]:
            best_type = "frustum"
            selection_info["strategy"] = "prefer_tapered_frustum_over_near_tie_cylinder"
    best_result = all_results[best_type]

    return {
        "best_type": best_type,
        "best_result": best_result,
        "scores": scores,
        "raw_scores": raw_scores,
        "filtered_out": filtered_out,
        "all_results": all_results,
        "fit_type_timings": fit_type_timings,
        "fit_search_total": time.perf_counter() - fit_search_start,
        "selection_info": selection_info,
    }

def create_o3d_bbox_lineset(corners: np.ndarray, color=(1, 0, 0)) -> o3d.geometry.LineSet:
    corners = np.asarray(corners, dtype=np.float64)
    lines = [
        [0,1],[0,2],[0,4],
        [1,3],[1,5],
        [2,3],[2,6],
        [3,7],
        [4,5],[4,6],
        [5,7],
        [6,7],
    ]
    colors = [list(color) for _ in lines]
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(corners)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls

def create_ellip_cylinder_mesh(
    center: np.ndarray,
    rotation: np.ndarray,
    semi_major: float,
    semi_minor: float,
    height: float,
    color=(0, 1, 0),
    resolution: int = 30,
) -> o3d.geometry.TriangleMesh:
    """创建椭圆柱 mesh。rotation 列 = [长轴, 短轴, 柱轴]。"""
    mesh = o3d.geometry.TriangleMesh.create_cylinder(
        radius=1.0, height=height, resolution=resolution,
    )
    verts = np.asarray(mesh.vertices).copy()
    verts[:, 0] *= semi_major
    verts[:, 1] *= semi_minor
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    mesh.rotate(rotation, center=np.zeros(3))
    mesh.translate(center)
    return mesh

def create_frustum_mesh(
    small_end_center: np.ndarray,
    rotation: np.ndarray,
    small_end_major: float,
    small_end_minor: float,
    large_end_major: float,
    large_end_minor: float,
    height: float,
    color=(0.1, 0.85, 0.45),
    resolution: int = 48,
    axial_steps: int = 18,
) -> o3d.geometry.TriangleMesh:
    """创建开口椭圆截锥 mesh。rotation 列 = [长轴, 短轴, 截锥轴]。"""
    vertices = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    triangles = []

    for i in range(axial_steps + 1):
        t = i / max(axial_steps, 1)
        major = (1.0 - t) * small_end_major + t * large_end_major
        minor = (1.0 - t) * small_end_minor + t * large_end_minor
        z = t * height
        for j in range(resolution):
            theta = 2.0 * np.pi * j / resolution
            u = major * np.cos(theta)
            v = minor * np.sin(theta)
            vertices.append(np.array([u, v, z], dtype=np.float64))

    first_ring = 1
    for j in range(resolution):
        j_next = (j + 1) % resolution
        triangles.append([0, first_ring + j_next, first_ring + j])

    for i in range(axial_steps):
        ring_start = 1 + i * resolution
        next_ring_start = 1 + (i + 1) * resolution
        for j in range(resolution):
            j_next = (j + 1) % resolution
            a = ring_start + j
            b = ring_start + j_next
            c = next_ring_start + j
            d = next_ring_start + j_next
            triangles.append([a, c, d])
            triangles.append([a, d, b])

    verts_local = np.asarray(vertices, dtype=np.float64)
    verts_world = verts_local @ rotation.T + small_end_center

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def create_bowl_mesh(
    vertex: np.ndarray,
    rotation: np.ndarray,
    rim_major: float,
    rim_minor: float,
    depth: float,
    color=(0.0, 0.7, 1.0),
    resolution: int = 48,
    radial_steps: int = 20,
) -> o3d.geometry.TriangleMesh:
    """创建开口椭圆抛物面 bowl mesh。rotation 列 = [长轴, 短轴, 碗轴]。"""
    vertices = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    triangles = []

    for i in range(1, radial_steps + 1):
        t = i / radial_steps
        w = depth * (t ** 2)
        for j in range(resolution):
            theta = 2.0 * np.pi * j / resolution
            u = t * rim_major * np.cos(theta)
            v = t * rim_minor * np.sin(theta)
            vertices.append(np.array([u, v, w], dtype=np.float64))

    first_ring = 1
    for j in range(resolution):
        j_next = (j + 1) % resolution
        triangles.append([0, first_ring + j, first_ring + j_next])

    for i in range(1, radial_steps):
        ring_start = 1 + (i - 1) * resolution
        next_ring_start = 1 + i * resolution
        for j in range(resolution):
            j_next = (j + 1) % resolution
            a = ring_start + j
            b = ring_start + j_next
            c = next_ring_start + j
            d = next_ring_start + j_next
            triangles.append([a, c, d])
            triangles.append([a, d, b])

    verts_local = np.asarray(vertices, dtype=np.float64)
    verts_world = verts_local @ rotation.T + vertex

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def create_plate_mesh(
    vertex: np.ndarray,
    rotation: np.ndarray,
    bottom_major: float,
    bottom_minor: float,
    rim_major: float,
    rim_minor: float,
    depth: float,
    color=(1.0, 0.65, 0.0),
    resolution: int = 48,
    radial_steps: int = 20,
) -> o3d.geometry.TriangleMesh:
    vertices = [np.array([0.0, 0.0, 0.0], dtype=np.float64)]
    triangles = []
    flat_ratio = 0.5 * (
        bottom_major / max(rim_major, 1e-8) +
        bottom_minor / max(rim_minor, 1e-8)
    )
    flat_ratio = float(np.clip(flat_ratio, 0.0, 0.99))

    for i in range(1, radial_steps + 1):
        t = i / radial_steps
        if t <= flat_ratio:
            local_depth = 0.0
        else:
            s = (t - flat_ratio) / max(1.0 - flat_ratio, 1e-8)
            local_depth = depth * float(_smoothstep01(np.array([s]))[0])
        for j in range(resolution):
            theta = 2.0 * np.pi * j / resolution
            u = t * rim_major * np.cos(theta)
            v = t * rim_minor * np.sin(theta)
            vertices.append(np.array([u, v, local_depth], dtype=np.float64))

    first_ring = 1
    for j in range(resolution):
        j_next = (j + 1) % resolution
        triangles.append([0, first_ring + j, first_ring + j_next])

    for i in range(1, radial_steps):
        ring_start = 1 + (i - 1) * resolution
        next_ring_start = 1 + i * resolution
        for j in range(resolution):
            j_next = (j + 1) % resolution
            a = ring_start + j
            b = ring_start + j_next
            c = next_ring_start + j
            d = next_ring_start + j_next
            triangles.append([a, c, d])
            triangles.append([a, d, b])

    verts_local = np.asarray(vertices, dtype=np.float64)
    verts_world = verts_local @ rotation.T + vertex

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_world)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh

def depth_to_pcd(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                 mask: Optional[np.ndarray] = None,
                 rgb: Optional[np.ndarray] = None) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    depth_map = np.asarray(depth, dtype=np.float64)
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    z = depth_map.copy()

    if mask is not None:
        z[mask == 0] = 0.0

    valid = np.isfinite(z) & (z > 0)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    xyz_map = np.full((H, W, 3), np.nan, dtype=np.float64)
    xyz_map[valid, 0] = x[valid]
    xyz_map[valid, 1] = y[valid]
    xyz_map[valid, 2] = z[valid]

    pts = np.stack((x[valid], y[valid], z[valid]), axis=-1)
    pcd = numpy_to_pcd(pts)

    if rgb is not None:
        rgb_img = np.asarray(rgb)
        if rgb_img.ndim == 3 and rgb_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        colors = rgb_img[valid].astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd, xyz_map

def cam_to_world(points: np.ndarray, R_cam2base: np.ndarray, t_cam2base: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    R = np.asarray(R_cam2base, dtype=np.float64)
    t = np.asarray(t_cam2base, dtype=np.float64).ravel()
    return (R @ points.T).T + t

def world_to_cam(points: np.ndarray, R_cam2base: np.ndarray, t_cam2base: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    R = np.asarray(R_cam2base, dtype=np.float64)
    t = np.asarray(t_cam2base, dtype=np.float64).ravel()
    return (points - t) @ R

def project_cam_to_image(
    points_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[np.ndarray, np.ndarray]:
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(f"points_cam must be (N,3), got {points_cam.shape}")

    uv = np.full((len(points_cam), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(points_cam).all(axis=1) & (points_cam[:, 2] > 1e-8)
    if np.any(valid):
        uv[valid, 0] = fx * points_cam[valid, 0] / points_cam[valid, 2] + cx
        uv[valid, 1] = fy * points_cam[valid, 1] / points_cam[valid, 2] + cy
    return uv, valid

def project_world_to_image(
    points_world: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    R_cam2base: np.ndarray,
    t_cam2base: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"points_world must be (N,3), got {points_world.shape}")
    points_cam = world_to_cam(points_world, R_cam2base, t_cam2base)
    uv, valid = project_cam_to_image(points_cam, fx, fy, cx, cy)
    return uv, points_cam, valid

def _estimate_projected_end_widths_from_mask(
    mask: np.ndarray,
    start_uv: np.ndarray,
    end_uv: np.ndarray,
) -> Tuple[Optional[float], Optional[float]]:
    mask = np.asarray(mask)
    if mask.ndim != 2:
        return None, None
    mask_bool = mask > 0
    ys, xs = np.nonzero(mask_bool)
    if len(xs) < 50:
        return None, None

    start_uv = np.asarray(start_uv, dtype=np.float64)
    end_uv = np.asarray(end_uv, dtype=np.float64)
    axis_2d = end_uv - start_uv
    axis_len = float(np.linalg.norm(axis_2d))
    if not np.isfinite(axis_len) or axis_len <= 1e-6:
        return None, None

    axis_dir = axis_2d / axis_len
    perp_dir = np.array([-axis_dir[1], axis_dir[0]], dtype=np.float64)
    pixels = np.column_stack([xs, ys]).astype(np.float64)
    rel = pixels - start_uv
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        t = (rel @ axis_dir) / axis_len
        d = rel @ perp_dir

    valid = np.isfinite(t) & np.isfinite(d) & (t >= -0.1) & (t <= 1.1)
    if int(valid.sum()) < 50:
        return None, None
    t = t[valid]
    d = d[valid]

    def _width_near(lo: float, hi: float) -> Optional[float]:
        sel = (t >= lo) & (t <= hi)
        if int(sel.sum()) < 20:
            return None
        return float(np.percentile(np.abs(d[sel]), 95.0) * 2.0)

    return _width_near(0.05, 0.20), _width_near(0.80, 0.95)

def _resolve_camera_frustum_end_orientation(
    frustum: FrustumFitResult,
    mask: Optional[np.ndarray],
    camera_intrinsics: Optional[np.ndarray],
) -> Tuple[FrustumFitResult, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "status": "skipped",
        "strategy": "mask_projected_width",
        "swapped": False,
    }
    if mask is None or camera_intrinsics is None:
        info["reason"] = "missing_mask_or_intrinsics"
        return frustum, info

    K = np.asarray(camera_intrinsics, dtype=np.float64)
    if K.shape != (3, 3):
        info["reason"] = "invalid_intrinsics_shape"
        return frustum, info

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    endpoints = np.vstack([
        np.asarray(frustum.small_end_center, dtype=np.float64),
        np.asarray(frustum.large_end_center, dtype=np.float64),
    ])
    uv, valid = project_cam_to_image(endpoints, fx, fy, cx, cy)
    if int(valid.sum()) != 2:
        info["reason"] = "endpoint_projection_invalid"
        return frustum, info

    small_width_px, large_width_px = _estimate_projected_end_widths_from_mask(
        mask=np.asarray(mask),
        start_uv=uv[0],
        end_uv=uv[1],
    )
    info.update({
        "small_end_uv": uv[0].tolist(),
        "large_end_uv": uv[1].tolist(),
        "small_end_width_px": small_width_px,
        "large_end_width_px": large_width_px,
    })
    if small_width_px is None or large_width_px is None:
        info["reason"] = "insufficient_mask_support"
        return frustum, info

    width_gap_px = float(small_width_px - large_width_px)
    width_ratio = float(small_width_px / max(large_width_px, 1e-6))
    info["width_gap_px"] = width_gap_px
    info["width_ratio"] = width_ratio

    has_visual_conflict = bool(width_gap_px > 3.0 and width_ratio > 1.05)
    info["visual_conflict"] = has_visual_conflict
    if not has_visual_conflict:
        info["status"] = "kept"
        info["reason"] = "projected_width_consistent_or_ambiguous"
        return frustum, info

    info["status"] = "visual_conflict"
    info["reason"] = "projected_small_end_appears_wider"
    return frustum, info

def _draw_text_panel(
    image_bgr: np.ndarray,
    lines: list[str],
    origin: Tuple[int, int] = (12, 12),
    font_scale: float = 0.44,
    line_gap: int = 6,
) -> None:
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    pad = 8
    x0, y0 = origin
    text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in lines]
    text_height = sum(size[1] for size in text_sizes) + line_gap * (len(lines) - 1)
    panel_w = max(size[0] for size in text_sizes) + 2 * pad
    panel_h = text_height + 2 * pad

    x1 = min(x0 + panel_w, image_bgr.shape[1] - 1)
    y1 = min(y0 + panel_h, image_bgr.shape[0] - 1)
    if x1 <= x0 or y1 <= y0:
        return

    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (20, 20, 20), thickness=-1)
    cv2.addWeighted(overlay, 0.28, image_bgr, 0.72, 0.0, dst=image_bgr)

    y = y0 + pad
    for line, size in zip(lines, text_sizes):
        y += size[1]
        cv2.putText(image_bgr, line, (x0 + pad, y), font, font_scale, (240, 240, 240), thickness, cv2.LINE_AA)
        y += line_gap

def _blend_overlay_layer(base_image: np.ndarray, overlay_image: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha <= 0.0:
        return np.asarray(base_image)
    if alpha >= 1.0:
        return np.asarray(overlay_image)
    blended = np.asarray(base_image).copy()
    cv2.addWeighted(np.asarray(overlay_image), alpha, np.asarray(base_image), 1.0 - alpha, 0.0, dst=blended)
    return blended

def save_cuboid_overlay_image(
    image_bgr: np.ndarray,
    object_points_cam: np.ndarray,
    cuboid: CuboidFitResult,
    K: np.ndarray,
    save_path: str,
) -> str:
    """将相机系点云、cuboid 线框和六个面四元数坐标轴投影到原图上。"""
    canvas = np.asarray(image_bgr).copy()
    H, W = canvas.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    object_points_cam = np.asarray(object_points_cam, dtype=np.float64)
    if len(object_points_cam) > 0:
        points_layer = canvas.copy()
        step = max(len(object_points_cam) // 2500, 1)
        sampled_points = object_points_cam[::step]
        uv_pts, valid_pts = project_cam_to_image(sampled_points, fx, fy, cx, cy)
        valid_pts &= (
            (uv_pts[:, 0] >= 0) & (uv_pts[:, 0] < W) &
            (uv_pts[:, 1] >= 0) & (uv_pts[:, 1] < H)
        )
        for u, v in uv_pts[valid_pts]:
            cv2.circle(points_layer, (int(round(u)), int(round(v))), 1, (255, 255, 0), thickness=-1)
        canvas = _blend_overlay_layer(canvas, points_layer, alpha=0.32)

    geom_layer = canvas.copy()

    bbox_edges = [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]
    corners_world = np.asarray(cuboid.corners_world, dtype=np.float64)
    uv_corners, valid_corners = project_cam_to_image(corners_world, fx, fy, cx, cy)
    for i, j in bbox_edges:
        if valid_corners[i] and valid_corners[j]:
            p0 = tuple(np.round(uv_corners[i]).astype(int))
            p1 = tuple(np.round(uv_corners[j]).astype(int))
            cv2.line(geom_layer, p0, p1, (0, 215, 255), thickness=1, lineType=cv2.LINE_AA)

    axis_len = float(np.clip(0.6 * np.min(cuboid.size), 0.02, 0.08))
    axis_specs = [
        ("x", (0, 0, 255), 0),
        ("y", (0, 255, 0), 1),
        ("z", (255, 0, 0), 2),
    ]

    face_order = ["x_plus", "x_minus", "y_plus", "y_minus", "z_plus", "z_minus"]
    for face_name in face_order:
        pose = cuboid.face_poses_wxyz[face_name]
        face_center = np.asarray(pose["position"], dtype=np.float64)
        quat = np.asarray(pose["quaternion_wxyz"], dtype=np.float64)
        face_R = quaternion_wxyz_to_rotation_matrix(quat)

        uv_center, valid_center = project_cam_to_image(face_center[None], fx, fy, cx, cy)
        if not valid_center[0]:
            continue
        center_px = tuple(np.round(uv_center[0]).astype(int))
        cv2.circle(geom_layer, center_px, 5, (255, 255, 255), thickness=-1)
        cv2.circle(geom_layer, center_px, 7, (0, 0, 0), thickness=1)

        label_anchor = np.array(center_px, dtype=np.int32)
        for axis_name, color, axis_idx in axis_specs:
            axis_end_world = face_center + axis_len * face_R[:, axis_idx]
            uv_axis, valid_axis = project_cam_to_image(
                np.vstack([face_center, axis_end_world]),
                fx, fy, cx, cy,
            )
            if not np.all(valid_axis):
                continue
            end_px = tuple(np.round(uv_axis[1]).astype(int))
            cv2.arrowedLine(
                geom_layer,
                center_px,
                end_px,
                color,
                thickness=2 if axis_name == "z" else 1,
                tipLength=0.22,
                line_type=cv2.LINE_AA,
            )
            if axis_name == "z":
                label_anchor = np.round(uv_axis[1]).astype(np.int32)

        label_pos = (int(label_anchor[0]) + 6, int(label_anchor[1]) - 6)
        cv2.putText(geom_layer, face_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(geom_layer, face_name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    canvas = _blend_overlay_layer(canvas, geom_layer, alpha=0.58)

    panel_lines = [
        "object points: cyan   cuboid: yellow",
        "quat axes: x/red  y/green  z/blue",
    ]
    for face_name in face_order:
        quat = np.asarray(cuboid.face_poses_wxyz[face_name]["quaternion_wxyz"], dtype=np.float64)
        panel_lines.append(
            f"{face_name}: q=[{quat[0]:+.2f},{quat[1]:+.2f},{quat[2]:+.2f},{quat[3]:+.2f}]"
        )
    _draw_text_panel(canvas, panel_lines, origin=(12, 12))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not cv2.imwrite(save_path, canvas):
        raise RuntimeError(f"failed to save overlay image to {save_path}")
    return os.path.abspath(save_path)

def _bbox_edges() -> list[Tuple[int, int]]:
    return [
        (0, 1), (0, 2), (0, 4),
        (1, 3), (1, 5),
        (2, 3), (2, 6),
        (3, 7),
        (4, 5), (4, 6),
        (5, 7),
        (6, 7),
    ]

def _sample_ellipse_ring_local(
    semi_major: float,
    semi_minor: float,
    z: float,
    resolution: int = 72,
) -> np.ndarray:
    thetas = np.linspace(0.0, 2.0 * np.pi, resolution + 1)
    return np.column_stack([
        semi_major * np.cos(thetas),
        semi_minor * np.sin(thetas),
        np.full_like(thetas, z),
    ]).astype(np.float64)

def _transform_local_polylines(
    polylines_local: list[np.ndarray],
    origin: np.ndarray,
    rotation: np.ndarray,
) -> list[np.ndarray]:
    origin = np.asarray(origin, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    return [poly @ rotation.T + origin for poly in polylines_local]

def _sample_cylinder_wireframe_world(
    cylinder: CylinderFitResult,
    ring_count: int = 5,
    longitudinal_count: int = 8,
    resolution: int = 72,
    longitudinal_steps: int = 24,
) -> list[np.ndarray]:
    polylines_local: list[np.ndarray] = []
    z_values = np.linspace(-0.5 * cylinder.height, 0.5 * cylinder.height, ring_count)
    for z in z_values:
        polylines_local.append(_sample_ellipse_ring_local(
            cylinder.semi_major, cylinder.semi_minor, float(z), resolution=resolution,
        ))

    thetas = np.linspace(0.0, 2.0 * np.pi, longitudinal_count, endpoint=False)
    z_line = np.linspace(-0.5 * cylinder.height, 0.5 * cylinder.height, longitudinal_steps)
    for theta in thetas:
        u = cylinder.semi_major * np.cos(theta)
        v = cylinder.semi_minor * np.sin(theta)
        polylines_local.append(np.column_stack([
            np.full_like(z_line, u),
            np.full_like(z_line, v),
            z_line,
        ]).astype(np.float64))

    return _transform_local_polylines(polylines_local, cylinder.center, cylinder.rotation)

def _sample_frustum_wireframe_world(
    frustum: FrustumFitResult,
    ring_count: int = 5,
    longitudinal_count: int = 8,
    resolution: int = 72,
    longitudinal_steps: int = 28,
) -> list[np.ndarray]:
    polylines_local: list[np.ndarray] = []
    t_rings = np.linspace(0.0, 1.0, ring_count)
    for t_ring in t_rings:
        major = (1.0 - t_ring) * frustum.small_end_major + t_ring * frustum.large_end_major
        minor = (1.0 - t_ring) * frustum.small_end_minor + t_ring * frustum.large_end_minor
        polylines_local.append(_sample_ellipse_ring_local(
            major, minor, t_ring * frustum.height, resolution=resolution,
        ))

    thetas = np.linspace(0.0, 2.0 * np.pi, longitudinal_count, endpoint=False)
    t_line = np.linspace(0.0, 1.0, longitudinal_steps)
    for theta in thetas:
        pts = []
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        for t_ring in t_line:
            major = (1.0 - t_ring) * frustum.small_end_major + t_ring * frustum.large_end_major
            minor = (1.0 - t_ring) * frustum.small_end_minor + t_ring * frustum.large_end_minor
            pts.append([
                major * cos_t,
                minor * sin_t,
                t_ring * frustum.height,
            ])
        polylines_local.append(np.asarray(pts, dtype=np.float64))

    return _transform_local_polylines(polylines_local, frustum.small_end_center, frustum.rotation)

def _sample_bowl_wireframe_world(
    bowl: BowlFitResult,
    ring_count: int = 5,
    longitudinal_count: int = 8,
    resolution: int = 72,
    longitudinal_steps: int = 28,
) -> list[np.ndarray]:
    polylines_local: list[np.ndarray] = []
    t_rings = np.linspace(0.2, 1.0, ring_count)
    for t_ring in t_rings:
        polylines_local.append(_sample_ellipse_ring_local(
            t_ring * bowl.rim_major,
            t_ring * bowl.rim_minor,
            bowl.depth * (t_ring ** 2),
            resolution=resolution,
        ))

    thetas = np.linspace(0.0, 2.0 * np.pi, longitudinal_count, endpoint=False)
    t_line = np.linspace(0.0, 1.0, longitudinal_steps)
    for theta in thetas:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        pts = []
        for t_ring in t_line:
            pts.append([
                t_ring * bowl.rim_major * cos_t,
                t_ring * bowl.rim_minor * sin_t,
                bowl.depth * (t_ring ** 2),
            ])
        polylines_local.append(np.asarray(pts, dtype=np.float64))

    return _transform_local_polylines(polylines_local, bowl.vertex, bowl.rotation)

def _sample_plate_wireframe_world(
    plate: PlateFitResult,
    ring_count: int = 5,
    longitudinal_count: int = 8,
    resolution: int = 72,
    longitudinal_steps: int = 28,
) -> list[np.ndarray]:
    polylines_local: list[np.ndarray] = []
    flat_ratio = 0.5 * (
        plate.bottom_major / max(plate.rim_major, 1e-8) +
        plate.bottom_minor / max(plate.rim_minor, 1e-8)
    )
    flat_ratio = float(np.clip(flat_ratio, 0.0, 0.99))

    def plate_depth_at(t_ring: float) -> float:
        if t_ring <= flat_ratio:
            return 0.0
        s = (t_ring - flat_ratio) / max(1.0 - flat_ratio, 1e-8)
        return plate.depth * float(_smoothstep01(np.array([s], dtype=np.float64))[0])

    t_rings = np.linspace(0.2, 1.0, ring_count)
    for t_ring in t_rings:
        polylines_local.append(_sample_ellipse_ring_local(
            t_ring * plate.rim_major,
            t_ring * plate.rim_minor,
            plate_depth_at(float(t_ring)),
            resolution=resolution,
        ))

    thetas = np.linspace(0.0, 2.0 * np.pi, longitudinal_count, endpoint=False)
    t_line = np.linspace(0.0, 1.0, longitudinal_steps)
    for theta in thetas:
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        pts = []
        for t_ring in t_line:
            pts.append([
                t_ring * plate.rim_major * cos_t,
                t_ring * plate.rim_minor * sin_t,
                plate_depth_at(float(t_ring)),
            ])
        polylines_local.append(np.asarray(pts, dtype=np.float64))

    return _transform_local_polylines(polylines_local, plate.vertex, plate.rotation)

def _sample_cuboid_wireframe_world(cuboid: CuboidFitResult) -> list[np.ndarray]:
    corners = np.asarray(cuboid.corners_world, dtype=np.float64)
    return [corners[[i, j]] for i, j in _bbox_edges()]

def _primitive_wireframe_world(fit_type: str, primitive: Any) -> list[np.ndarray]:
    if fit_type == "cuboid":
        return _sample_cuboid_wireframe_world(primitive)
    if fit_type == "cylinder":
        return _sample_cylinder_wireframe_world(primitive)
    if fit_type == "frustum":
        return _sample_frustum_wireframe_world(primitive)
    if fit_type == "bowl":
        return _sample_bowl_wireframe_world(primitive)
    if fit_type == "plate":
        return _sample_plate_wireframe_world(primitive)
    raise ValueError(f"Unsupported primitive type for wireframe: {fit_type}")

def _primitive_obb_corners_world(fit_type: str, primitive: Any) -> np.ndarray:
    if fit_type == "cuboid":
        return np.asarray(primitive.corners_world, dtype=np.float64)
    if fit_type == "cylinder":
        max_local = np.array([
            primitive.semi_major,
            primitive.semi_minor,
            0.5 * primitive.height,
        ], dtype=np.float64)
        return transform_local_to_world(
            cuboid_corners_from_bounds(-max_local, max_local),
            primitive.center,
            primitive.rotation,
        )
    if fit_type == "frustum":
        half_extents = np.array([
            max(primitive.small_end_major, primitive.large_end_major),
            max(primitive.small_end_minor, primitive.large_end_minor),
            primitive.height,
        ], dtype=np.float64)
        min_local = np.array([-half_extents[0], -half_extents[1], 0.0], dtype=np.float64)
        max_local = np.array([half_extents[0], half_extents[1], half_extents[2]], dtype=np.float64)
        return transform_local_to_world(
            cuboid_corners_from_bounds(min_local, max_local),
            primitive.small_end_center,
            primitive.rotation,
        )
    if fit_type == "bowl":
        min_local = np.array([-primitive.rim_major, -primitive.rim_minor, 0.0], dtype=np.float64)
        max_local = np.array([primitive.rim_major, primitive.rim_minor, primitive.depth], dtype=np.float64)
        return transform_local_to_world(
            cuboid_corners_from_bounds(min_local, max_local),
            primitive.vertex,
            primitive.rotation,
        )
    if fit_type == "plate":
        min_local = np.array([-primitive.rim_major, -primitive.rim_minor, 0.0], dtype=np.float64)
        max_local = np.array([primitive.rim_major, primitive.rim_minor, primitive.depth], dtype=np.float64)
        return transform_local_to_world(
            cuboid_corners_from_bounds(min_local, max_local),
            primitive.vertex,
            primitive.rotation,
        )
    raise ValueError(f"Unsupported primitive type for OBB: {fit_type}")

def _draw_projected_polyline(
    image_bgr: np.ndarray,
    points_cam: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if len(points_cam) < 2:
        return

    uv, valid = project_cam_to_image(points_cam, fx, fy, cx, cy)
    for idx in range(len(uv) - 1):
        if not (valid[idx] and valid[idx + 1]):
            continue
        p0 = tuple(np.round(uv[idx]).astype(int))
        p1 = tuple(np.round(uv[idx + 1]).astype(int))
        cv2.line(image_bgr, p0, p1, color, thickness=thickness, lineType=cv2.LINE_AA)

def _primitive_overlay_panel_lines(fit_type: str, primitive: Any) -> list[str]:
    lines = [
        f"fit: {fit_type}   residual: {primitive.residual:.5f}",
        "fit overlay: green   obb: orange   object points: cyan",
    ]
    if fit_type in {"frustum", "cylinder"}:
        lines.append("axis: white   small_end: magenta   large_end: yellow")
    elif fit_type in {"bowl", "plate"}:
        lines.append("axis: white   start: magenta   end: yellow")
    if fit_type == "cuboid":
        size = np.asarray(primitive.size, dtype=np.float64)
        lines.append(f"size = [{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]")
    elif fit_type == "cylinder":
        lines.append(
            f"a/b = {primitive.semi_major:.3f}/{primitive.semi_minor:.3f}   h = {primitive.height:.3f}"
        )
    elif fit_type == "frustum":
        lines.append(
            "small_end = "
            f"{primitive.small_end_major:.3f}/{primitive.small_end_minor:.3f}   "
            "large_end = "
            f"{primitive.large_end_major:.3f}/{primitive.large_end_minor:.3f}   "
            f"h = {primitive.height:.3f}"
        )
        if primitive.pose_type is not None:
            lines.append(f"pose = {primitive.pose_type}")
    elif fit_type == "bowl":
        lines.append(
            f"rim = {primitive.rim_major:.3f}/{primitive.rim_minor:.3f}   depth = {primitive.depth:.3f}"
        )
        if primitive.pose_type is not None:
            lines.append(f"pose = {primitive.pose_type}")
    elif fit_type == "plate":
        lines.append(
            "bottom = "
            f"{primitive.bottom_major:.3f}/{primitive.bottom_minor:.3f}   "
            "rim = "
            f"{primitive.rim_major:.3f}/{primitive.rim_minor:.3f}   "
            f"depth = {primitive.depth:.3f}"
        )
        if primitive.pose_type is not None:
            lines.append(f"pose = {primitive.pose_type}   flat = {primitive.bottom_flatness:.4f}")
        else:
            lines.append(f"flat = {primitive.bottom_flatness:.4f}")
    return lines

def _primitive_axis_endpoints_cam(
    fit_type: str,
    primitive: Any,
) -> Optional[Tuple[np.ndarray, np.ndarray, str, str]]:
    if fit_type == "frustum":
        return (
            np.asarray(primitive.small_end_center, dtype=np.float64),
            np.asarray(primitive.large_end_center, dtype=np.float64),
            "small_end",
            "large_end",
        )
    if fit_type == "cylinder":
        return (
            np.asarray(primitive.bottom_center, dtype=np.float64),
            np.asarray(primitive.top_center, dtype=np.float64),
            "small_end",
            "large_end",
        )
    if fit_type == "bowl":
        return (
            np.asarray(primitive.vertex, dtype=np.float64),
            np.asarray(primitive.rim_center, dtype=np.float64),
            "vertex",
            "rim",
        )
    if fit_type == "plate":
        return (
            np.asarray(primitive.vertex, dtype=np.float64),
            np.asarray(primitive.rim_center, dtype=np.float64),
            "center",
            "rim",
        )
    return None


def render_primitive_overlay_image(
    image_bgr: np.ndarray,
    object_points_cam: np.ndarray,
    fit_type: str,
    primitive: Any,
    K: np.ndarray,
) -> np.ndarray:
    """将相机系下的拟合基元线框和同轴 OBB 投影到原图上，并直接返回图像。"""
    canvas = np.asarray(image_bgr).copy()
    H, W = canvas.shape[:2]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    object_points_cam = np.asarray(object_points_cam, dtype=np.float64)
    if len(object_points_cam) > 0:
        points_layer = canvas.copy()
        step = max(len(object_points_cam) // 2500, 1)
        sampled_points = object_points_cam[::step]
        uv_pts, valid_pts = project_cam_to_image(sampled_points, fx, fy, cx, cy)
        valid_pts &= (
            (uv_pts[:, 0] >= 0) & (uv_pts[:, 0] < W) &
            (uv_pts[:, 1] >= 0) & (uv_pts[:, 1] < H)
        )
        for u, v in uv_pts[valid_pts]:
            cv2.circle(points_layer, (int(round(u)), int(round(v))), 1, (255, 255, 0), thickness=-1)
        canvas = _blend_overlay_layer(canvas, points_layer, alpha=0.32)

    geom_layer = canvas.copy()

    for poly_world in _primitive_wireframe_world(fit_type, primitive):
        _draw_projected_polyline(
            geom_layer,
            poly_world,
            fx, fy, cx, cy,
            color=(80, 255, 80),
            thickness=1,
        )

    obb_corners = _primitive_obb_corners_world(fit_type, primitive)
    uv_corners, valid_corners = project_cam_to_image(obb_corners, fx, fy, cx, cy)
    for i, j in _bbox_edges():
        if not (valid_corners[i] and valid_corners[j]):
            continue
        p0 = tuple(np.round(uv_corners[i]).astype(int))
        p1 = tuple(np.round(uv_corners[j]).astype(int))
        cv2.line(geom_layer, p0, p1, (0, 165, 255), thickness=1, lineType=cv2.LINE_AA)

    axis_endpoints = _primitive_axis_endpoints_cam(fit_type, primitive)
    if axis_endpoints is not None:
        start_cam, end_cam, start_label, end_label = axis_endpoints
        uv_axis, valid_axis = project_cam_to_image(
            np.vstack([start_cam, end_cam]),
            fx, fy, cx, cy,
        )
        if bool(np.all(valid_axis)):
            start_px = tuple(np.round(uv_axis[0]).astype(int))
            end_px = tuple(np.round(uv_axis[1]).astype(int))
            cv2.arrowedLine(
                geom_layer,
                start_px,
                end_px,
                (255, 255, 255),
                thickness=1,
                tipLength=0.18,
                line_type=cv2.LINE_AA,
            )
            cv2.circle(geom_layer, start_px, 5, (255, 0, 255), thickness=-1)
            cv2.circle(geom_layer, end_px, 5, (0, 255, 255), thickness=-1)
            cv2.putText(geom_layer, start_label, (start_px[0] + 6, start_px[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(geom_layer, start_label, (start_px[0] + 6, start_px[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(geom_layer, end_label, (end_px[0] + 6, end_px[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(geom_layer, end_label, (end_px[0] + 6, end_px[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

    canvas = _blend_overlay_layer(canvas, geom_layer, alpha=0.58)
    _draw_text_panel(canvas, _primitive_overlay_panel_lines(fit_type, primitive), origin=(12, 12))
    return canvas

def save_primitive_overlay_image(
    image_bgr: np.ndarray,
    object_points_cam: np.ndarray,
    fit_type: str,
    primitive: Any,
    K: np.ndarray,
    save_path: str,
) -> str:
    """将相机系下的拟合基元线框和同轴 OBB 投影到原图上。"""
    canvas = render_primitive_overlay_image(
        image_bgr=image_bgr,
        object_points_cam=object_points_cam,
        fit_type=fit_type,
        primitive=primitive,
        K=K,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if not cv2.imwrite(save_path, canvas):
        raise RuntimeError(f"failed to save overlay image to {save_path}")
    return os.path.abspath(save_path)

def _save_fit_visualization(
    obj_pcd: o3d.geometry.PointCloud,
    primitive_geom: Union[o3d.geometry.LineSet, o3d.geometry.TriangleMesh],
    save_dir: str,
    scene_pcd: Optional[o3d.geometry.PointCloud] = None,
    scene_voxel_size: float = 0.005,
    primitive_filename: Optional[str] = None,
) -> Dict[str, str]:
    """导出可视化 PLY。

    combined.ply : 场景点云（降采样、浅灰）与物体点云（高亮蓝）合并在一起，
                   场景稀疏化后不会完全遮挡物体。
    bbox.ply / cylinder.ply / frustum.ply / bowl.ply : 拟合基元，在 MeshLab 里叠加 combined.ply 查看。
    """
    os.makedirs(save_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    merged = o3d.geometry.PointCloud()
    if scene_pcd is not None:
        scene_down = scene_pcd.voxel_down_sample(voxel_size=scene_voxel_size)
        if not scene_down.has_colors():
            scene_down.paint_uniform_color([0.78, 0.78, 0.78])
        merged += scene_down
    obj_copy = o3d.geometry.PointCloud(obj_pcd)
    obj_copy.paint_uniform_color([1.0, 0.2, 0.2])
    merged += obj_copy

    combined_path = os.path.join(save_dir, "combined.ply")
    o3d.io.write_point_cloud(combined_path, merged)
    paths["combined"] = os.path.abspath(combined_path)

    if isinstance(primitive_geom, o3d.geometry.LineSet):
        prim_path = os.path.join(save_dir, primitive_filename or "bbox.ply")
        o3d.io.write_line_set(prim_path, primitive_geom)
    else:
        prim_path = os.path.join(save_dir, primitive_filename or "cylinder.ply")
        o3d.io.write_triangle_mesh(prim_path, primitive_geom)
    paths["primitive"] = os.path.abspath(prim_path)
    return paths

def fit_object_primitive(
    object_points_cam: np.ndarray,
    robust_percentile: float = 1.0,
    cluster_eps: float = 0.02,
    cluster_min_points: int = 30,
    enable_outlier_removal: bool = True,
    fit_type: Optional[str] = None,
    vis_save_dir: Optional[str] = None,
    scene_pcd_cam: Optional[o3d.geometry.PointCloud] = None,
    scene_voxel_size: float = 0.005,
    image_bgr: Optional[np.ndarray] = None,
    image_mask: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    fit_object_start = time.perf_counter()
    preprocess: Dict[str, Any] = {
        "input_points": int(len(object_points_cam)),
        "largest_cluster_points": int(len(object_points_cam)),
        "outlier_removal_enabled": bool(enable_outlier_removal),
    }
    timings: Dict[str, Any] = {}
    pcd = numpy_to_pcd(object_points_cam)
    cluster_start = time.perf_counter()
    pcd = keep_largest_cluster(pcd, eps=cluster_eps, min_points=cluster_min_points)
    timings["largest_cluster"] = time.perf_counter() - cluster_start
    preprocess["largest_cluster_points"] = int(len(pcd.points))
    if enable_outlier_removal:
        outlier_start = time.perf_counter()
        pcd, outlier_stats = remove_point_outliers(pcd)
        timings["outlier_removal"] = time.perf_counter() - outlier_start
        preprocess["outlier_removal"] = outlier_stats

    pts = pcd_to_numpy(pcd)
    preprocess["final_points"] = int(len(pts))
    if len(pts) < 20:
        raise ValueError("聚类后物体点云过少 (<20)，无法拟合。")

    fit_search_start = time.perf_counter()
    result = fit_best_primitive(
        pts,
        robust_percentile=robust_percentile,
        fit_type=fit_type,
    )
    timings["fit_best_primitive"] = time.perf_counter() - fit_search_start
    timings["fit_search_total"] = result.get("fit_search_total")
    timings["fit_type_timings"] = result.get("fit_type_timings", {})
    frustum_end_resolution = None
    if "frustum" in result.get("all_results", {}):
        resolved_frustum, frustum_end_resolution = _resolve_camera_frustum_end_orientation(
            frustum=result["all_results"]["frustum"],
            mask=image_mask,
            camera_intrinsics=camera_intrinsics,
        )
        result["all_results"]["frustum"] = resolved_frustum
        if result.get("best_type") == "frustum":
            result["best_result"] = resolved_frustum
    result = {**result, "preprocess": preprocess, "frame": "camera", "timings": timings}
    if frustum_end_resolution is not None:
        result["frustum_end_resolution"] = frustum_end_resolution

    if vis_save_dir is not None:
        vis_start = time.perf_counter()
        obj_pcd = numpy_to_pcd(pts)
        primitive_filename = "primitive.ply"

        if result["best_type"] == "cuboid":
            box = result["best_result"]
            primitive_geom = create_o3d_bbox_lineset(box.corners_world, color=(1, 0, 0))
            primitive_filename = "bbox.ply"
        elif result["best_type"] == "cylinder":
            cyl = result["best_result"]
            primitive_geom = create_ellip_cylinder_mesh(
                center=cyl.center, rotation=cyl.rotation,
                semi_major=cyl.semi_major, semi_minor=cyl.semi_minor,
                height=cyl.height, color=(0, 1, 0),
            )
            primitive_filename = "cylinder.ply"
        elif result["best_type"] == "frustum":
            frustum = result["best_result"]
            primitive_geom = create_frustum_mesh(
                small_end_center=frustum.small_end_center,
                rotation=frustum.rotation,
                small_end_major=frustum.small_end_major,
                small_end_minor=frustum.small_end_minor,
                large_end_major=frustum.large_end_major,
                large_end_minor=frustum.large_end_minor,
                height=frustum.height,
                color=(0.1, 0.85, 0.45),
            )
            primitive_filename = "frustum.ply"
        elif result["best_type"] == "plate":
            plate = result["best_result"]
            primitive_geom = create_plate_mesh(
                vertex=plate.vertex,
                rotation=plate.rotation,
                bottom_major=plate.bottom_major,
                bottom_minor=plate.bottom_minor,
                rim_major=plate.rim_major,
                rim_minor=plate.rim_minor,
                depth=plate.depth,
                color=(1.0, 0.65, 0.0),
            )
            primitive_filename = "plate.ply"
        else:
            bowl = result["best_result"]
            primitive_geom = create_bowl_mesh(
                vertex=bowl.vertex,
                rotation=bowl.rotation,
                rim_major=bowl.rim_major,
                rim_minor=bowl.rim_minor,
                depth=bowl.depth,
                color=(0.0, 0.7, 1.0),
            )
            primitive_filename = "bowl.ply"

        saved = _save_fit_visualization(
            obj_pcd, primitive_geom, vis_save_dir,
            scene_pcd=scene_pcd_cam, scene_voxel_size=scene_voxel_size,
            primitive_filename=primitive_filename,
        )
        if image_bgr is not None and camera_intrinsics is not None:
            overlay_path = os.path.join(vis_save_dir, f"left_{result['best_type']}_overlay.png")
            saved["image_overlay"] = save_primitive_overlay_image(
                image_bgr=image_bgr,
                object_points_cam=pts,
                fit_type=result["best_type"],
                primitive=result["best_result"],
                K=np.asarray(camera_intrinsics, dtype=np.float64),
                save_path=overlay_path,
            )
        result = {**result, "vis_saved_paths": saved}
        timings["visualization"] = time.perf_counter() - vis_start
    timings["fit_object_total"] = time.perf_counter() - fit_object_start
    return result


def fit_camera_primitive_from_points(
    object_points_cam: np.ndarray,
    robust_percentile: float = 1.0,
    fit_type: Optional[str] = None,
    image_mask: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Fit the best primitive directly from preprocessed camera-frame object points.

    This is the lightweight entry point for other modules that already have a cleaned
    object point cloud and only need primitive fitting plus optional frustum end
    disambiguation from the 2D mask.
    """
    pts = ensure_points(object_points_cam)
    if len(pts) < 20:
        raise ValueError("object point cloud too small for primitive fitting (<20)")

    result = fit_best_primitive(
        pts,
        robust_percentile=robust_percentile,
        fit_type=fit_type,
    )
    result = {**result, "frame": "camera"}
    if "frustum" in result.get("all_results", {}):
        resolved_frustum, frustum_end_resolution = _resolve_camera_frustum_end_orientation(
            frustum=result["all_results"]["frustum"],
            mask=image_mask,
            camera_intrinsics=camera_intrinsics,
        )
        result["all_results"]["frustum"] = resolved_frustum
        result["frustum_end_resolution"] = frustum_end_resolution
        if result.get("best_type") == "frustum":
            result["best_result"] = resolved_frustum
    return result

def create_scene_pcd(points: np.ndarray, colors: Optional[np.ndarray] = None) -> o3d.geometry.PointCloud:
    pcd = numpy_to_pcd(points)
    if colors is not None:
        colors = np.asarray(colors, dtype=np.float64)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError(f"colors must be (N,3), got {colors.shape}")
        if len(colors) != len(points):
            raise ValueError("scene colors must have the same length as scene points")
        if colors.size > 0 and np.nanmax(colors) > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def _to_serializable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj

def serialize_fit_result(result: Dict[str, Any]) -> Dict[str, Any]:
    return _to_serializable(result)


def summarize_fit_result_lines(
    fit_result: Optional[Dict[str, Any]],
    prefix: Optional[str] = None,
) -> list[str]:
    if not fit_result:
        return [f"{prefix}: unavailable" if prefix else "unavailable"]

    def _fetch(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    fit_type = fit_result.get("best_type", "?")
    best = fit_result.get("best_result")
    header = f"{prefix}: {fit_type}" if prefix else f"type: {fit_type}"
    lines = [header]
    residual = _fetch(best, "residual")
    if residual is not None:
        lines.append(f"residual={float(residual):.5f}")
    scores = fit_result.get("scores") or {}
    if scores:
        ordered = sorted(((str(k), float(v)) for k, v in scores.items()), key=lambda kv: kv[1])[:3]
        lines.append("scores=" + ", ".join(f"{k}:{v:.5f}" for k, v in ordered))

    if fit_type == "cuboid":
        size = np.asarray(_fetch(best, "size", [0.0, 0.0, 0.0]), dtype=np.float64)
        lines.append(f"size=[{size[0]:.3f}, {size[1]:.3f}, {size[2]:.3f}]m")
    elif fit_type == "cylinder":
        lines.append(
            "a/b/h="
            f"{float(_fetch(best, 'semi_major', 0.0)):.3f}/"
            f"{float(_fetch(best, 'semi_minor', 0.0)):.3f}/"
            f"{float(_fetch(best, 'height', 0.0)):.3f}m"
        )
    elif fit_type == "frustum":
        lines.append(
            "small="
            f"{float(_fetch(best, 'small_end_major', 0.0)):.3f}/"
            f"{float(_fetch(best, 'small_end_minor', 0.0)):.3f}  "
            "large="
            f"{float(_fetch(best, 'large_end_major', 0.0)):.3f}/"
            f"{float(_fetch(best, 'large_end_minor', 0.0)):.3f}  "
            f"h={float(_fetch(best, 'height', 0.0)):.3f}m"
        )
        info = fit_result.get("frustum_end_resolution") or {}
        if info:
            lines.append(
                "frustum_end="
                f"{info.get('status')} "
                f"small_w={info.get('small_end_width_px')} "
                f"large_w={info.get('large_end_width_px')}"
            )
    elif fit_type == "bowl":
        lines.append(
            "rim/depth="
            f"{float(_fetch(best, 'rim_major', 0.0)):.3f}/"
            f"{float(_fetch(best, 'rim_minor', 0.0)):.3f}/"
            f"{float(_fetch(best, 'depth', 0.0)):.3f}m"
        )
    elif fit_type == "plate":
        lines.append(
            "bottom="
            f"{float(_fetch(best, 'bottom_major', 0.0)):.3f}/"
            f"{float(_fetch(best, 'bottom_minor', 0.0)):.3f}  "
            "rim="
            f"{float(_fetch(best, 'rim_major', 0.0)):.3f}/"
            f"{float(_fetch(best, 'rim_minor', 0.0)):.3f}  "
            f"depth={float(_fetch(best, 'depth', 0.0)):.3f}m"
        )
    return lines


def render_best_fit_overlay_image(
    image_bgr: np.ndarray,
    object_points_cam: np.ndarray,
    fit_result: Dict[str, Any],
    K: np.ndarray,
) -> np.ndarray:
    best_type = fit_result["best_type"]
    best_result = fit_result["best_result"]
    return render_primitive_overlay_image(
        image_bgr=image_bgr,
        object_points_cam=object_points_cam,
        fit_type=best_type,
        primitive=best_result,
        K=K,
    )

def _format_seconds(seconds: Any) -> str:
    if seconds is None:
        return "n/a"
    return f"{float(seconds):.4f}s"

def print_fit_result_summary(fit_result: Dict[str, Any]) -> None:
    print(f"\n===== 拟合结果 =====")
    frame = fit_result.get("frame", "camera")
    frame_label = "camera" if frame == "camera" else str(frame)
    preprocess = fit_result.get("preprocess", {})
    if preprocess:
        print("点云预处理:")
        print(
            f"  input={preprocess.get('input_points', 'n/a')} -> "
            f"largest_cluster={preprocess.get('largest_cluster_points', 'n/a')} -> "
            f"final={preprocess.get('final_points', 'n/a')}"
        )
        outlier = preprocess.get("outlier_removal")
        if outlier:
            print(
                f"  outlier_removal: "
                f"{outlier.get('initial_points', 'n/a')} -> "
                f"{outlier.get('after_statistical', 'n/a')} -> "
                f"{outlier.get('after_radius', 'n/a')}"
            )
    pipeline_timing = fit_result.get("pipeline_timing", {})
    if pipeline_timing:
        print("主流程耗时:")
        for stage_name, seconds in pipeline_timing.items():
            print(f"  {stage_name}: {_format_seconds(seconds)}")
    timings = fit_result.get("timings", {})
    if timings:
        print("拟合阶段耗时:")
        for key in ("largest_cluster", "outlier_removal", "fit_best_primitive", "fit_search_total", "visualization", "fit_object_total"):
            if key in timings:
                print(f"  {key}: {_format_seconds(timings[key])}")
        fit_type_timings = timings.get("fit_type_timings", {})
        if fit_type_timings:
            print("各类型拟合耗时:")
            for fit_name, seconds in fit_type_timings.items():
                print(f"  {fit_name}: {_format_seconds(seconds)}")
    print(f"最佳类型: {fit_result['best_type']}")
    print(f"参与评选残差: {fit_result['scores']}")
    if "raw_scores" in fit_result:
        print(f"原始残差:     {fit_result['raw_scores']}")
    selection_info = fit_result.get("selection_info")
    if selection_info:
        strategy = selection_info.get("strategy")
        if strategy:
            print(f"选择策略:     {strategy}")
        frustum_vs_cylinder = selection_info.get("frustum_cylinder_resolution")
        if frustum_vs_cylinder:
            print(
                "frustum/cylinder判定: "
                f"prefer_frustum={frustum_vs_cylinder.get('prefer_frustum')}, "
                f"residual_gap={frustum_vs_cylinder.get('residual_gap', 0.0):.6f}, "
                f"relative_gap={frustum_vs_cylinder.get('relative_gap', 0.0):.3f}, "
                f"taper_ratio={frustum_vs_cylinder.get('taper_ratio', 0.0):.3f}, "
                f"taper_delta={frustum_vs_cylinder.get('taper_delta', 0.0):.4f}"
            )
    frustum_end_resolution = fit_result.get("frustum_end_resolution")
    if frustum_end_resolution:
        print(
            "frustum端点修正: "
            f"status={frustum_end_resolution.get('status')}, "
            f"swapped={frustum_end_resolution.get('swapped')}, "
            f"small_w={frustum_end_resolution.get('small_end_width_px')}, "
            f"large_w={frustum_end_resolution.get('large_end_width_px')}, "
            f"reason={frustum_end_resolution.get('reason')}"
        )
    filtered_out = fit_result.get("filtered_out", {})
    if filtered_out:
        print("已过滤类型:")
        for fit_name, info in filtered_out.items():
            status = info.get("status", "unknown")
            reason = info.get("reason", "")
            if "raw_residual" in info:
                print(f"  {fit_name}: status={status}, raw_residual={info['raw_residual']:.6f}, reason={reason}")
            else:
                print(f"  {fit_name}: status={status}, reason={reason}")

    if fit_result["best_type"] == "cuboid":
        box = fit_result["best_result"]
        print(f"  center ({frame_label}): {box['center']}")
        print(f"  size:           {box['size']}")
        print("  face_poses_wxyz:")
        for face_name, pose in box["face_poses_wxyz"].items():
            print(f"    {face_name}: pos={pose['position']}, quat={pose['quaternion_wxyz']}")
        print(f"  residual:       {box['residual']:.6f}")
    elif fit_result["best_type"] == "cylinder":
        cyl = fit_result["best_result"]
        print(f"  center ({frame_label}): {cyl['center']}")
        print(f"  axis:           {cyl['axis']}")
        print(f"  semi_major:     {cyl['semi_major']:.4f}")
        print(f"  semi_minor:     {cyl['semi_minor']:.4f}")
        print(f"  height:         {cyl['height']:.4f}")
        print(f"  residual:       {cyl['residual']:.6f}")
    elif fit_result["best_type"] == "frustum":
        frustum = fit_result["best_result"]
        print(f"  center ({frame_label}): {frustum['center']}")
        print(f"  small_end_center:{frustum['small_end_center']}")
        print(f"  large_end_center:{frustum['large_end_center']}")
        print(f"  axis:           {frustum['axis']}")
        print(f"  small_end_major:{frustum['small_end_major']:.4f}")
        print(f"  small_end_minor:{frustum['small_end_minor']:.4f}")
        print(f"  large_end_major:{frustum['large_end_major']:.4f}")
        print(f"  large_end_minor:{frustum['large_end_minor']:.4f}")
        print(f"  height:         {frustum['height']:.4f}")
        if frustum.get("pose_type") is not None:
            print(f"  pose_type:      {frustum['pose_type']}")
        print(f"  residual:       {frustum['residual']:.6f}")
    elif fit_result["best_type"] == "plate":
        plate = fit_result["best_result"]
        print(f"  vertex ({frame_label}): {plate['vertex']}")
        print(f"  rim_center:     {plate['rim_center']}")
        print(f"  axis:           {plate['axis']}")
        print(f"  bottom_major:   {plate['bottom_major']:.4f}")
        print(f"  bottom_minor:   {plate['bottom_minor']:.4f}")
        print(f"  rim_major:      {plate['rim_major']:.4f}")
        print(f"  rim_minor:      {plate['rim_minor']:.4f}")
        print(f"  depth:          {plate['depth']:.4f}")
        if plate.get("pose_type") is not None:
            print(f"  pose_type:      {plate['pose_type']}")
        print(f"  flatness:       {plate['bottom_flatness']:.6f}")
        print(f"  residual:       {plate['residual']:.6f}")
    else:
        bowl = fit_result["best_result"]
        print(f"  vertex ({frame_label}): {bowl['vertex']}")
        print(f"  rim_center:     {bowl['rim_center']}")
        print(f"  axis:           {bowl['axis']}")
        print(f"  rim_major:      {bowl['rim_major']:.4f}")
        print(f"  rim_minor:      {bowl['rim_minor']:.4f}")
        print(f"  depth:          {bowl['depth']:.4f}")
        if bowl.get("pose_type") is not None:
            print(f"  pose_type:      {bowl['pose_type']}")
        print(f"  residual:       {bowl['residual']:.6f}")

    if "vis_saved_paths" in fit_result:
        print("可视化文件:")
        for key, path in fit_result["vis_saved_paths"].items():
            print(f"  {key}: {path}")

def run_demo(
    obj_name: Optional[str],
    point_coords: Optional[np.ndarray],
    prefix: str,
    data_root: str,
    idx: int,
    s2m2_url: str,
    sam3_url: str,
    fit_type: Optional[str],
    vis_save_dir: str,
    fx: float = DEFAULT_FX,
    fy: float = DEFAULT_FY,
    cx: float = DEFAULT_CX,
    cy: float = DEFAULT_CY,
    baseline: float = DEFAULT_BASELINE,
) -> Dict[str, Any]:
    run_demo_start = time.perf_counter()
    pipeline_timing: Dict[str, float] = {}
    os.makedirs(os.path.join(data_root, "images", "head_clouds2m2_depth"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "images", "head_mask"), exist_ok=True)
    os.makedirs(os.path.join(data_root, "images", "head_masked"), exist_ok=True)

    left_path = f"{data_root}/00_inputs/left_pcd.png"
    right_path = f"{data_root}/00_inputs/right_pcd.png"
    depth_path = f"{data_root}/00_inputs/depth.npy"
    mask_path = f"{data_root}/00_inputs/sam_mask.png"
    masked_path = f"{data_root}/00_inputs/masked.png"
    os.makedirs(vis_save_dir, exist_ok=True)

    fx = float(fx)
    fy = float(fy)
    cx = float(cx)
    cy = float(cy)
    baseline = float(baseline)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    depth_stage_start = time.perf_counter()
    if not os.path.exists(depth_path):
        K_str = " ".join(str(K[i, j]) for i in range(3) for j in range(3))
        files = {"left_file": open(left_path, "rb"), "right_file": open(right_path, "rb")}
        data = {"K": K_str, "baseline": np.float32(baseline)}
        r = requests.post(s2m2_url, files=files, data=data, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(f"S2M2 request failed: {r.status_code} {r.text}")
        xyz_map = np.load(io.BytesIO(r.content))
        depth = xyz_map[:, :, 2]
        np.save(depth_path, depth)
        print("Saved:", depth_path)
        files["left_file"].close()
        files["right_file"].close()
    else:
        depth = np.load(depth_path)
    pipeline_timing["depth_io"] = time.perf_counter() - depth_stage_start

    mask_stage_start = time.perf_counter()
    if not os.path.exists(mask_path):
        if obj_name is not None and point_coords is not None:
            mask = request_sam3_mask(
                image_path=left_path,
                text_prompt=obj_name,
                point_prompts=[[point_coords[0], point_coords[1]]],
                point_labels=[1],
                server_url=sam3_url,
            )
        elif point_coords is not None:
            mask = request_sam3_mask(
                image_path=left_path,
                point_prompts=[[point_coords[0], point_coords[1]]],
                point_labels=[1],
                server_url=sam3_url,
            )
        else:
            mask = request_sam3_mask(
                image_path=left_path,
                text_prompt=obj_name,
                server_url=sam3_url,
            )
        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        cv2.imwrite(mask_path, mask)
        print("Saved:", mask_path)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    pipeline_timing["mask_io"] = time.perf_counter() - mask_stage_start

    image_stage_start = time.perf_counter()
    left_rgb = cv2.imread(left_path)
    masked_rgb = cv2.bitwise_and(left_rgb, left_rgb, mask=mask)
    if not os.path.exists(masked_path):
        cv2.imwrite(masked_path, masked_rgb)
        print("Saved:", masked_path)
    pipeline_timing["image_prepare"] = time.perf_counter() - image_stage_start

    pcd_stage_start = time.perf_counter()
    scene_pcd_cam, _ = depth_to_pcd(depth, fx, fy, cx, cy, rgb=left_rgb)
    scene_pts_cam = pcd_to_numpy(scene_pcd_cam)
    scene_colors = np.asarray(scene_pcd_cam.colors) if scene_pcd_cam.has_colors() else None

    obj_pcd_cam, _ = depth_to_pcd(depth, fx, fy, cx, cy, mask=mask)
    obj_pts_cam = pcd_to_numpy(obj_pcd_cam)
    pipeline_timing["pointcloud_build"] = time.perf_counter() - pcd_stage_start

    print(f"物体点云: {len(obj_pts_cam)} 点 (相机坐标系)")
    print(f"  x 范围: [{obj_pts_cam[:,0].min():.4f}, {obj_pts_cam[:,0].max():.4f}]")
    print(f"  y 范围: [{obj_pts_cam[:,1].min():.4f}, {obj_pts_cam[:,1].max():.4f}]")
    print(f"  z 范围: [{obj_pts_cam[:,2].min():.4f}, {obj_pts_cam[:,2].max():.4f}]")

    scene_pcd_cam = create_scene_pcd(scene_pts_cam, scene_colors)
    fit_stage_start = time.perf_counter()
    fit_result = fit_object_primitive(
        object_points_cam=obj_pts_cam,
        fit_type=fit_type,
        vis_save_dir=vis_save_dir,
        scene_pcd_cam=scene_pcd_cam,
        image_bgr=left_rgb,
        image_mask=mask,
        camera_intrinsics=K,
    )
    pipeline_timing["fit_pipeline"] = time.perf_counter() - fit_stage_start
    fit_result = serialize_fit_result(fit_result)
    pipeline_timing["run_demo_total"] = time.perf_counter() - run_demo_start
    fit_result["pipeline_timing"] = pipeline_timing

    print_fit_result_summary(fit_result)
    return fit_result

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj_name", default="cup", type=str, help="object name")
    parser.add_argument("--point_coords", default=None, type=float, nargs=2,
                        metavar=("X", "Y"), help="point coordinates, e.g. --point_coords 530 440")
    parser.add_argument("--data_root", default="/Users/tianyizhang/codes/pcd-fit/test/test-0")

    parser.add_argument("--idx", default=0, type=int)
    parser.add_argument("--s2m2_url", default="http://101.132.143.105:5060/api/process")
    parser.add_argument("--sam3_url", default=SAM3_SERVER_URL)
    parser.add_argument("--fx", default=DEFAULT_FX, type=float)
    parser.add_argument("--fy", default=DEFAULT_FY, type=float)
    parser.add_argument("--cx", default=DEFAULT_CX, type=float)
    parser.add_argument("--cy", default=DEFAULT_CY, type=float)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE, type=float)
    parser.add_argument("--fit_type", default=None, choices=["cuboid", "cylinder", "frustum", "bowl", "plate"])
    parser.add_argument("--vis_save_dir", default=None, type=str)
    args = parser.parse_args()

    obj_name = None
    point_coords = None
    if args.obj_name is None and args.point_coords is None:
        raise ValueError("obj_name and point_coords cannot be both None")
    elif args.obj_name is not None and args.point_coords is not None:
        obj_name = args.obj_name
        point_coords = np.array(args.point_coords)
        prefix = f"{obj_name}-{point_coords[0]:.2f}-{point_coords[1]:.2f}"
    elif args.obj_name is not None:
        obj_name = args.obj_name
        prefix = obj_name
    elif args.point_coords is not None:
        point_coords = np.array(args.point_coords)
        prefix = f"point-{point_coords[0]:.2f}-{point_coords[1]:.2f}"

    vis_save_dir = args.vis_save_dir or f"./vis_in_cam_results/{prefix}-{args.idx:06d}"
    run_demo(
        obj_name=obj_name,
        point_coords=point_coords,
        prefix=prefix,
        data_root=args.data_root,
        idx=args.idx,
        s2m2_url=args.s2m2_url,
        sam3_url=args.sam3_url,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        baseline=args.baseline,
        fit_type=args.fit_type,
        vis_save_dir=vis_save_dir,
    )

if __name__ == "__main__":
    main()
