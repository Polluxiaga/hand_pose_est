#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

DEFAULT_TARGET_KIND = "auto"
SUPPORTED_TARGET_KINDS = (
    "auto",
    "support_surface",
    "top_open_box",
    "front_open_cabinet",
    "closed_box",
)
SUPPORTED_PRIMITIVE_SHAPES = ("plane", "box", "cylinder")

DEFAULT_SUPPORT_CLEARANCE_M = 0.004
DEFAULT_EDGE_CLEARANCE_M = 0.010
DEFAULT_CONTAINER_WALL_CLEARANCE_M = 0.020
DEFAULT_CONTAINER_RIM_CLEARANCE_M = 0.008
DEFAULT_SCENE_COLLISION_MARGIN_M = 0.004
DEFAULT_ARM_CLEARANCE_MAX_DISTANCE_M = 0.45
DEFAULT_ARM_CLEARANCE_RADIUS_M = 0.055
DEFAULT_GRIPPER_FORWARD_MIN_BASE_X = 0.0
# Soft tiebreak only: hard orientation filtering happens before scoring.
DEFAULT_X_DOWN_SCORE_WEIGHT = 6.4
DEFAULT_GRIPPER_X_MAX_BASE_Z = 0.0
DEFAULT_CANDIDATE_YAW_SWEEP_COUNT = 12
DEFAULT_PICK_TO_PLACE_ROTATION_SCORE_WEIGHT = 2.4
DEFAULT_PICK_TO_PLACE_TRANSLATION_SCORE_WEIGHT = 0.8
DEFAULT_AS_PICKED_GRASP_SCORE_WEIGHT = 0.9
DEFAULT_SUPPORT_COVERAGE_SCORE_WEIGHT = 1.1
DEFAULT_FACE_BAND_RATIO = 0.18
DEFAULT_FACE_VISIBLE_MIN_COS = 0.05
DEFAULT_MAX_COORD_ABS = 1e6
DEFAULT_FRAME_YAW_SEARCH_STEPS = 90
DEFAULT_RELAXED_TOP_OPEN_MIN_SLENDER_RATIO = 2.0
DEFAULT_RELAXED_TOP_OPEN_TILT_ANGLES_DEG = (20.0, 35.0, 50.0)
DEFAULT_SUPPORT_OFFSET_FRACTIONS = (0.0, -0.22, 0.22, -0.42, 0.42)


def _empty_points() -> np.ndarray:
    return np.zeros((0, 3), dtype=np.float32)


def _finite_point_mask(points: np.ndarray, max_abs: float = DEFAULT_MAX_COORD_ABS) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return np.zeros((0,), dtype=bool)
    keep = np.all(np.isfinite(pts), axis=1)
    keep &= np.max(np.abs(pts), axis=1) <= float(max_abs)
    return keep.astype(bool, copy=False)


def _as_points(points: np.ndarray | None) -> np.ndarray:
    pts = np.asarray(points if points is not None else _empty_points(), dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        return _empty_points()
    finite = _finite_point_mask(pts)
    if not np.any(finite):
        return _empty_points()
    return pts[finite].astype(np.float32, copy=False)


def _is_finite_rotation(rotation: np.ndarray) -> bool:
    rot = np.asarray(rotation, dtype=np.float64)
    return bool(rot.shape == (3, 3) and np.all(np.isfinite(rot)))


def _wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(float(angle)), math.cos(float(angle)))


def _rot2(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _yaw_to_rotation(yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    unit = _normalize(np.asarray(axis, dtype=np.float64).reshape(3), fallback=np.array([0.0, 0.0, 1.0], dtype=np.float64))
    x, y, z = float(unit[0]), float(unit[1]), float(unit[2])
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    ident = np.eye(3, dtype=np.float64)
    s = math.sin(float(angle_rad))
    c = math.cos(float(angle_rad))
    return ident + s * skew + (1.0 - c) * (skew @ skew)


def _normalize(vec: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size > 0 and np.all(np.isfinite(arr)):
        norm = float(np.linalg.norm(arr))
        if norm > 1e-8:
            return (arr / norm).astype(np.float64, copy=False)
    if fallback is not None:
        fb = np.asarray(fallback, dtype=np.float64).reshape(-1)
        if fb.size > 0 and np.all(np.isfinite(fb)):
            norm = float(np.linalg.norm(fb))
            if norm > 1e-8:
                return (fb / norm).astype(np.float64, copy=False)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _normalize_xy(vec_xy: np.ndarray, fallback_xy: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(vec_xy, dtype=np.float64).reshape(-1)
    if arr.size >= 2 and np.all(np.isfinite(arr[:2])):
        norm = float(np.linalg.norm(arr[:2]))
        if norm > 1e-8:
            return (arr[:2] / norm).astype(np.float64, copy=False)
    if fallback_xy is not None:
        fb = np.asarray(fallback_xy, dtype=np.float64).reshape(-1)
        if fb.size >= 2 and np.all(np.isfinite(fb[:2])):
            norm = float(np.linalg.norm(fb[:2]))
            if norm > 1e-8:
                return (fb[:2] / norm).astype(np.float64, copy=False)
    return np.array([1.0, 0.0], dtype=np.float64)


def _rotation_to_yaw(rotation: np.ndarray) -> float:
    rot = np.asarray(rotation, dtype=np.float64)
    if rot.shape != (3, 3):
        return 0.0
    return float(math.atan2(float(rot[1, 0]), float(rot[0, 0])))


def _rotation_geodesic_angle(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
    rot_a = np.asarray(rotation_a, dtype=np.float64)
    rot_b = np.asarray(rotation_b, dtype=np.float64)
    if rot_a.shape != (3, 3) or rot_b.shape != (3, 3):
        return float(math.pi)
    rel = rot_a.T @ rot_b
    cos_theta = 0.5 * (float(np.trace(rel)) - 1.0)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(math.acos(cos_theta))


def _unique_angles(angles_rad: list[float], tol_rad: float = math.radians(1.0)) -> list[float]:
    out: list[float] = []
    for angle in angles_rad:
        wrapped = _wrap_to_pi(float(angle))
        if all(abs(_wrap_to_pi(wrapped - prev)) > float(tol_rad) for prev in out):
            out.append(wrapped)
    return out


def _world_to_local(points_world: np.ndarray, origin_world: np.ndarray, rotation_world: np.ndarray) -> np.ndarray:
    pts = _as_points(points_world).astype(np.float64, copy=False)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    origin = np.asarray(origin_world, dtype=np.float64).reshape(1, 3)
    rot = np.asarray(rotation_world, dtype=np.float64)
    if not np.all(np.isfinite(origin)) or not _is_finite_rotation(rot):
        return np.zeros((0, 3), dtype=np.float64)
    return (pts - origin) @ rot


def _local_to_world(points_local: np.ndarray, origin_world: np.ndarray, rotation_world: np.ndarray) -> np.ndarray:
    pts = _as_points(points_local).astype(np.float64, copy=False)
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    origin = np.asarray(origin_world, dtype=np.float64).reshape(1, 3)
    rot = np.asarray(rotation_world, dtype=np.float64)
    if not np.all(np.isfinite(origin)) or not _is_finite_rotation(rot):
        return np.zeros((0, 3), dtype=np.float64)
    return pts @ rot.T + origin


def _box_corners_world(center_world: np.ndarray, rotation_world: np.ndarray, size_xyz: np.ndarray) -> np.ndarray:
    ctr = np.asarray(center_world, dtype=np.float64).reshape(3)
    rot = np.asarray(rotation_world, dtype=np.float64)
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    half = 0.5 * size
    local = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
            [-half[0], -half[1], half[2]],
            [half[0], -half[1], half[2]],
            [half[0], half[1], half[2]],
            [-half[0], half[1], half[2]],
        ],
        dtype=np.float64,
    )
    return _local_to_world(local, ctr, rot).astype(np.float32, copy=False)


def _project_points_to_camera(
    points_base: np.ndarray,
    constraint: CameraVisibilityConstraint,
    min_depth_m: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = _as_points(points_base).astype(np.float64, copy=False)
    if pts.shape[0] == 0 or not constraint.available:
        return (
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=bool),
        )
    pts_cam = (pts - constraint.shift_cb.astype(np.float64).reshape((1, 3))) @ constraint.rot_cb.astype(np.float64)
    depth = pts_cam[:, 2]
    valid_depth = np.isfinite(depth) & (depth > float(min_depth_m))
    u = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    v = np.full((pts.shape[0],), np.nan, dtype=np.float64)
    fx = float(constraint.k1[0, 0])
    fy = float(constraint.k1[1, 1])
    cx = float(constraint.k1[0, 2])
    cy = float(constraint.k1[1, 2])
    u[valid_depth] = fx * pts_cam[valid_depth, 0] / depth[valid_depth] + cx
    v[valid_depth] = fy * pts_cam[valid_depth, 1] / depth[valid_depth] + cy
    return u, v, valid_depth


def _points_in_camera_frame_mask(
    points_base: np.ndarray,
    constraint: CameraVisibilityConstraint,
    min_depth_m: float = 1e-4,
) -> tuple[np.ndarray, dict[str, Any]]:
    u, v, valid_depth = _project_points_to_camera(points_base, constraint, min_depth_m=min_depth_m)
    width, height = constraint.image_size_wh
    margin = max(0, int(constraint.pixel_margin_px))
    in_frame = (
        valid_depth
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= float(margin))
        & (u <= float(width - 1 - margin))
        & (v >= float(margin))
        & (v <= float(height - 1 - margin))
    )
    return in_frame.astype(bool, copy=False), {
        "valid_depth_count": int(np.count_nonzero(valid_depth)),
        "in_frame_count": int(np.count_nonzero(in_frame)),
        "all_valid_depth": bool(np.all(valid_depth)) if valid_depth.size > 0 else False,
        "all_in_frame": bool(np.all(in_frame)) if in_frame.size > 0 else False,
    }


def _evaluate_candidate_camera_visibility(
    object_box_place: OrientedBox,
    gripper_pose_place: FramePose,
    constraint: CameraVisibilityConstraint,
) -> dict[str, Any]:
    if not constraint.available:
        return {"available": False}
    object_points = object_box_place.corners_world().astype(np.float32, copy=False)
    grasp_points = np.asarray(gripper_pose_place.origin_base, dtype=np.float32).reshape(1, 3)
    object_in_frame, object_debug = _points_in_camera_frame_mask(object_points, constraint)
    grasp_in_frame, grasp_debug = _points_in_camera_frame_mask(grasp_points, constraint)
    object_ok = bool(object_debug.get("all_in_frame", False))
    grasp_ok = bool(grasp_debug.get("all_in_frame", False))
    overall_ok = (
        (not constraint.require_full_object_in_frame or object_ok)
        and (not constraint.require_grasp_origin_in_frame or grasp_ok)
    )
    return {
        "available": True,
        "all_in_frame": bool(overall_ok),
        "object": dict(object_debug),
        "grasp": dict(grasp_debug),
        "object_in_frame_mask": object_in_frame.astype(bool, copy=False).tolist(),
        "grasp_in_frame_mask": grasp_in_frame.astype(bool, copy=False).tolist(),
    }


def _safe_quantile(values: np.ndarray, q: float, default: float) -> float:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, float(q)))


def _projected_half_extent(size_xy: np.ndarray, yaw_delta_rad: float) -> np.ndarray:
    sx = 0.5 * float(size_xy[0])
    sy = 0.5 * float(size_xy[1])
    c = abs(math.cos(float(yaw_delta_rad)))
    s = abs(math.sin(float(yaw_delta_rad)))
    return np.array([c * sx + s * sy, s * sx + c * sy], dtype=np.float64)


def _slender_metrics(size_xyz: np.ndarray) -> tuple[int, np.ndarray, float]:
    dims = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    order = np.argsort(dims)
    longest_axis = int(order[-1])
    second = float(dims[order[-2]])
    slender_ratio = float(dims[longest_axis] / max(second, 1e-4))
    return longest_axis, dims, slender_ratio


def _sample_object_box_volume_world(object_box: OrientedBox) -> np.ndarray:
    longest_axis, dims, slender_ratio = _slender_metrics(object_box.size_xyz)
    counts = np.array([3, 3, 3], dtype=np.int32)
    counts[longest_axis] = 7 if slender_ratio >= DEFAULT_RELAXED_TOP_OPEN_MIN_SLENDER_RATIO else 5
    axes = [
        np.linspace(-0.5 * float(dims[idx]), 0.5 * float(dims[idx]), int(counts[idx]), dtype=np.float64)
        for idx in range(3)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    grid = np.stack(mesh, axis=-1).reshape(-1, 3)
    corners = _box_corners_world(
        np.zeros((3,), dtype=np.float64),
        np.eye(3, dtype=np.float64),
        dims.astype(np.float64, copy=False),
    ).astype(np.float64, copy=False)
    local_points = np.concatenate(
        [
            np.zeros((1, 3), dtype=np.float64),
            grid,
            corners,
        ],
        axis=0,
    )
    return _local_to_world(local_points, object_box.center_base, object_box.rotation_base).astype(np.float32, copy=False)


def _sample_object_box_bottom_face_world(object_box: OrientedBox) -> np.ndarray:
    dims = np.asarray(object_box.size_xyz, dtype=np.float64).reshape(3)
    half = 0.5 * dims
    planar_dims = dims[:2]
    planar_order = np.argsort(planar_dims)
    major_axis = int(planar_order[-1])
    minor_axis = int(planar_order[0])
    planar_slender_ratio = float(planar_dims[major_axis] / max(planar_dims[minor_axis], 1e-4))
    counts = np.array([5, 5], dtype=np.int32)
    counts[major_axis] = 11 if planar_slender_ratio >= 6.0 else (9 if planar_slender_ratio >= 3.0 else 7)
    counts[minor_axis] = 3 if planar_slender_ratio >= 3.0 else 5
    axes = [
        np.linspace(-half[idx], half[idx], int(counts[idx]), dtype=np.float64)
        for idx in range(2)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    face_grid = np.stack(mesh, axis=-1).reshape(-1, 2)
    local_points = np.column_stack(
        [
            face_grid,
            np.full((face_grid.shape[0],), -half[2], dtype=np.float64),
        ]
    )
    bottom_corners = np.array(
        [
            [-half[0], -half[1], -half[2]],
            [half[0], -half[1], -half[2]],
            [half[0], half[1], -half[2]],
            [-half[0], half[1], -half[2]],
        ],
        dtype=np.float64,
    )
    local_points = np.concatenate(
        [
            np.array([[0.0, 0.0, -half[2]]], dtype=np.float64),
            local_points,
            bottom_corners,
        ],
        axis=0,
    )
    return _local_to_world(local_points, object_box.center_base, object_box.rotation_base).astype(np.float32, copy=False)


def _support_surface_local_inside_mask(
    profile: "TargetProfile",
    points_local: np.ndarray,
    margin_m: float = 0.0,
) -> np.ndarray:
    pts = np.asarray(points_local, dtype=np.float64).reshape((-1, 3))
    if pts.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    margin = max(0.0, float(margin_m))
    if str(profile.primitive_shape) == "cylinder":
        radius = max(
            1e-4,
            float(profile.primitive_radius or 0.5 * min(float(profile.size_xyz[0]), float(profile.size_xyz[1])))
            - margin,
        )
        return (np.linalg.norm(pts[:, :2], axis=1) <= (radius + 1e-6)).astype(bool, copy=False)
    half_target = 0.5 * np.asarray(profile.size_xyz[:2], dtype=np.float64).reshape(2)
    bounds = np.maximum(half_target - margin, 1e-4)
    inside = (np.abs(pts[:, 0]) <= (bounds[0] + 1e-6)) & (np.abs(pts[:, 1]) <= (bounds[1] + 1e-6))
    return inside.astype(bool, copy=False)


def _support_surface_min_coverage(object_box_place: OrientedBox, profile: "TargetProfile") -> float:
    _, dims, slender_ratio = _slender_metrics(object_box_place.size_xyz)
    min_coverage = 0.52
    if slender_ratio >= 6.0:
        min_coverage = 0.30
    elif slender_ratio >= 3.0:
        min_coverage = 0.38
    support_xy = np.asarray(profile.size_xyz[:2], dtype=np.float64).reshape(2)
    obj_xy = np.sort(np.asarray(dims[:2], dtype=np.float64))
    if obj_xy[0] <= 0.40 * max(float(np.min(support_xy)), 1e-4):
        min_coverage = min(min_coverage, 0.40)
    return float(np.clip(min_coverage, 0.20, 0.75))


def _support_surface_center_offsets_local(
    profile: "TargetProfile",
    preferred_center_base: np.ndarray | None,
) -> list[tuple[str, np.ndarray]]:
    base_xy = np.zeros((2,), dtype=np.float64)
    if preferred_center_base is not None:
        preferred_local = profile.world_to_local(
            np.asarray(preferred_center_base, dtype=np.float32).reshape(1, 3)
        ).reshape(3)
        base_xy = np.asarray(preferred_local[:2], dtype=np.float64)

    if str(profile.primitive_shape) == "cylinder":
        radius = max(
            1e-4,
            float(profile.primitive_radius or 0.5 * min(float(profile.size_xyz[0]), float(profile.size_xyz[1]))),
        )
        base_norm = float(np.linalg.norm(base_xy))
        if base_norm > radius and base_norm > 1e-6:
            base_xy = base_xy / base_norm * radius
        return [("center", base_xy.astype(np.float64, copy=False))]

    half_target = 0.5 * np.asarray(profile.size_xyz[:2], dtype=np.float64).reshape(2)

    def _axis_values(base_val: float, half_extent: float) -> list[float]:
        vals = [float(base_val)]
        for frac in DEFAULT_SUPPORT_OFFSET_FRACTIONS[1:]:
            candidate = float(base_val + frac * half_extent)
            vals.append(float(np.clip(candidate, -half_extent, half_extent)))
        unique: list[float] = []
        for val in vals:
            if all(abs(val - prev) > 1e-4 for prev in unique):
                unique.append(float(val))
        return unique

    xs = _axis_values(float(base_xy[0]), float(half_target[0]))
    ys = _axis_values(float(base_xy[1]), float(half_target[1]))
    ordered: list[tuple[str, np.ndarray]] = []
    seen: list[np.ndarray] = []
    combos = [(x, y) for x in xs for y in ys]
    combos.sort(key=lambda xy: float((xy[0] - base_xy[0]) ** 2 + (xy[1] - base_xy[1]) ** 2))
    for x_val, y_val in combos:
        candidate_xy = np.array([float(x_val), float(y_val)], dtype=np.float64)
        if any(np.max(np.abs(candidate_xy - prev)) <= 1e-4 for prev in seen):
            continue
        seen.append(candidate_xy)
        if float(np.linalg.norm(candidate_xy - base_xy)) <= 1e-4:
            label = "center"
        else:
            label = f"offset=({candidate_xy[0]:+.3f},{candidate_xy[1]:+.3f})m"
        ordered.append((label, candidate_xy))
    return ordered


@dataclass
class FramePose:
    origin_base: np.ndarray
    rotation_base: np.ndarray

    def __post_init__(self) -> None:
        self.origin_base = np.asarray(self.origin_base, dtype=np.float32).reshape(3)
        rot = np.asarray(self.rotation_base, dtype=np.float64).reshape(3, 3)
        x_axis = _normalize(rot[:, 0], fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
        y_seed = rot[:, 1]
        z_axis = rot[:, 2] - x_axis * float(np.dot(rot[:, 2], x_axis))
        z_axis = _normalize(z_axis, fallback=np.cross(x_axis, y_seed))
        y_axis = np.cross(z_axis, x_axis)
        y_axis = _normalize(y_axis, fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64))
        z_axis = _normalize(np.cross(x_axis, y_axis), fallback=z_axis)
        self.rotation_base = np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32, copy=False)

    @property
    def x_axis_base(self) -> np.ndarray:
        return self.rotation_base[:, 0].astype(np.float32, copy=False)

    @property
    def y_axis_base(self) -> np.ndarray:
        return self.rotation_base[:, 1].astype(np.float32, copy=False)

    @property
    def z_axis_base(self) -> np.ndarray:
        return self.rotation_base[:, 2].astype(np.float32, copy=False)

    def copy(self) -> "FramePose":
        return FramePose(
            origin_base=self.origin_base.astype(np.float32, copy=True),
            rotation_base=self.rotation_base.astype(np.float32, copy=True),
        )


@dataclass
class OrientedBox:
    center_base: np.ndarray
    rotation_base: np.ndarray
    size_xyz: np.ndarray

    def __post_init__(self) -> None:
        self.center_base = np.asarray(self.center_base, dtype=np.float32).reshape(3)
        self.rotation_base = FramePose(
            origin_base=np.zeros((3,), dtype=np.float32),
            rotation_base=np.asarray(self.rotation_base, dtype=np.float32).reshape(3, 3),
        ).rotation_base
        self.size_xyz = np.maximum(np.asarray(self.size_xyz, dtype=np.float32).reshape(3), 1e-4)

    @property
    def yaw_rad(self) -> float:
        return _rotation_to_yaw(self.rotation_base)

    @property
    def top_center_base(self) -> np.ndarray:
        return (
            self.center_base.astype(np.float64)
            + self.rotation_base[:, 2].astype(np.float64) * (0.5 * float(self.size_xyz[2]))
        ).astype(np.float32)

    def corners_world(self) -> np.ndarray:
        return _box_corners_world(self.center_base, self.rotation_base, self.size_xyz)


@dataclass
class PlannerConfig:
    support_clearance_m: float = DEFAULT_SUPPORT_CLEARANCE_M
    edge_clearance_m: float = DEFAULT_EDGE_CLEARANCE_M
    container_wall_clearance_m: float = DEFAULT_CONTAINER_WALL_CLEARANCE_M
    container_rim_clearance_m: float = DEFAULT_CONTAINER_RIM_CLEARANCE_M
    scene_collision_margin_m: float = DEFAULT_SCENE_COLLISION_MARGIN_M
    arm_clearance_max_distance_m: float = DEFAULT_ARM_CLEARANCE_MAX_DISTANCE_M
    arm_clearance_radius_m: float = DEFAULT_ARM_CLEARANCE_RADIUS_M
    gripper_forward_min_base_x: float = DEFAULT_GRIPPER_FORWARD_MIN_BASE_X
    x_down_score_weight: float = DEFAULT_X_DOWN_SCORE_WEIGHT
    gripper_x_max_base_z: float = DEFAULT_GRIPPER_X_MAX_BASE_Z
    candidate_yaw_sweep_count: int = DEFAULT_CANDIDATE_YAW_SWEEP_COUNT
    pick_to_place_rotation_score_weight: float = DEFAULT_PICK_TO_PLACE_ROTATION_SCORE_WEIGHT
    pick_to_place_translation_score_weight: float = DEFAULT_PICK_TO_PLACE_TRANSLATION_SCORE_WEIGHT
    as_picked_grasp_score_weight: float = DEFAULT_AS_PICKED_GRASP_SCORE_WEIGHT
    support_coverage_score_weight: float = DEFAULT_SUPPORT_COVERAGE_SCORE_WEIGHT


@dataclass
class CameraVisibilityConstraint:
    k1: np.ndarray
    rot_cb: np.ndarray
    shift_cb: np.ndarray
    image_size_wh: tuple[int, int]
    pixel_margin_px: int = 0
    require_full_object_in_frame: bool = True
    require_grasp_origin_in_frame: bool = True

    def __post_init__(self) -> None:
        self.k1 = np.asarray(self.k1, dtype=np.float32).reshape(3, 3)
        self.rot_cb = np.asarray(self.rot_cb, dtype=np.float32).reshape(3, 3)
        self.shift_cb = np.asarray(self.shift_cb, dtype=np.float32).reshape(3)
        width = int(self.image_size_wh[0]) if len(tuple(self.image_size_wh)) >= 1 else 0
        height = int(self.image_size_wh[1]) if len(tuple(self.image_size_wh)) >= 2 else 0
        self.image_size_wh = (max(0, width), max(0, height))
        self.pixel_margin_px = max(0, int(self.pixel_margin_px))
        self.require_full_object_in_frame = bool(self.require_full_object_in_frame)
        self.require_grasp_origin_in_frame = bool(self.require_grasp_origin_in_frame)

    @property
    def available(self) -> bool:
        width, height = self.image_size_wh
        fx = float(self.k1[0, 0])
        fy = float(self.k1[1, 1])
        return bool(
            width > 0
            and height > 0
            and _is_finite_rotation(self.rot_cb)
            and np.all(np.isfinite(self.shift_cb))
            and np.all(np.isfinite(self.k1))
            and abs(fx) > 1e-8
            and abs(fy) > 1e-8
        )


@dataclass
class TargetProfile:
    kind: str
    requested_kind: str
    primitive_shape: str
    center_base: np.ndarray
    rotation_base: np.ndarray
    size_xyz: np.ndarray
    primitive_radius: float | None
    top_z: float
    bottom_z: float
    floor_z: float
    opening_face: str | None
    opening_normal_base: np.ndarray | None
    top_open_score: float
    front_open_score: float
    face_densities: dict[str, float]
    points_base: np.ndarray
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        primitive = str(self.primitive_shape or "box").strip().lower()
        self.primitive_shape = primitive if primitive in SUPPORTED_PRIMITIVE_SHAPES else "box"
        self.center_base = np.asarray(self.center_base, dtype=np.float32).reshape(3)
        self.rotation_base = FramePose(
            origin_base=np.zeros((3,), dtype=np.float32),
            rotation_base=np.asarray(self.rotation_base, dtype=np.float32).reshape(3, 3),
        ).rotation_base
        self.size_xyz = np.maximum(np.asarray(self.size_xyz, dtype=np.float32).reshape(3), 1e-4)
        if self.primitive_radius is not None:
            radius = float(self.primitive_radius)
            self.primitive_radius = radius if np.isfinite(radius) and radius > 1e-5 else None
        self.points_base = _as_points(self.points_base)
        if self.opening_normal_base is not None:
            self.opening_normal_base = _normalize(
                np.asarray(self.opening_normal_base, dtype=np.float64).reshape(3),
                fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64),
            ).astype(np.float32, copy=False)

    @property
    def yaw_rad(self) -> float:
        return _rotation_to_yaw(self.rotation_base)

    def world_to_local(self, points_world: np.ndarray) -> np.ndarray:
        return _world_to_local(points_world, self.center_base, self.rotation_base).astype(np.float32, copy=False)

    def local_to_world(self, points_local: np.ndarray) -> np.ndarray:
        return _local_to_world(points_local, self.center_base, self.rotation_base).astype(np.float32, copy=False)


@dataclass
class PlannerInput:
    object_box_current: OrientedBox
    pick_pose_base: FramePose
    target_points_base: np.ndarray
    scene_collision_points_base: np.ndarray
    target_kind_hint: str = DEFAULT_TARGET_KIND
    table_points_base: np.ndarray = field(default_factory=_empty_points)
    preferred_center_base: np.ndarray | None = None
    target_profile_override: TargetProfile | None = None
    start_object_center_base: np.ndarray | None = None
    start_gripper_axes_base: np.ndarray | None = None
    camera_visibility_constraint: CameraVisibilityConstraint | None = None
    allow_best_effort: bool = True
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = str(self.target_kind_hint or DEFAULT_TARGET_KIND).strip().lower()
        self.target_kind_hint = kind if kind in SUPPORTED_TARGET_KINDS else DEFAULT_TARGET_KIND
        self.target_points_base = _as_points(self.target_points_base)
        self.scene_collision_points_base = _as_points(self.scene_collision_points_base)
        self.table_points_base = _as_points(self.table_points_base)
        if self.preferred_center_base is not None:
            center = np.asarray(self.preferred_center_base, dtype=np.float32).reshape(-1)
            if center.shape[0] >= 3 and np.all(np.isfinite(center[:3])):
                self.preferred_center_base = center[:3].astype(np.float32, copy=False)
            else:
                self.preferred_center_base = None
        if self.start_object_center_base is not None:
            center = np.asarray(self.start_object_center_base, dtype=np.float32).reshape(-1)
            if center.shape[0] >= 3 and np.all(np.isfinite(center[:3])):
                self.start_object_center_base = center[:3].astype(np.float32, copy=False)
            else:
                self.start_object_center_base = None
        if self.start_gripper_axes_base is not None:
            axes = np.asarray(self.start_gripper_axes_base, dtype=np.float32)
            if axes.shape == (3, 3) and np.all(np.isfinite(axes)):
                self.start_gripper_axes_base = FramePose(
                    origin_base=np.zeros((3,), dtype=np.float32),
                    rotation_base=axes,
                ).rotation_base
            else:
                self.start_gripper_axes_base = None
        self.allow_best_effort = bool(self.allow_best_effort)


@dataclass
class PlacementCandidate:
    rule_id: str
    object_box_place: OrientedBox
    gripper_pose_place: FramePose
    score: float
    score_terms: dict[str, float]
    notes: list[str]
    valid: bool = True


@dataclass
class PlacementPlanResult:
    success: bool
    best_candidate: PlacementCandidate | None
    all_candidates: list[PlacementCandidate]
    target_profile: TargetProfile
    selected_rule_id: str | None
    debug: dict[str, Any] = field(default_factory=dict)


class PlacementRule(Protocol):
    rule_id: str

    def match_score(self, profile: TargetProfile, planner_input: PlannerInput) -> float:
        ...

    def generate_candidates(
        self,
        profile: TargetProfile,
        planner_input: PlannerInput,
        object_to_gripper_origin_local: np.ndarray,
        object_to_gripper_rotation_local: np.ndarray,
        config: PlannerConfig,
    ) -> list[PlacementCandidate]:
        ...


def _default_pick_long_axis(object_box: OrientedBox) -> np.ndarray:
    rot = np.asarray(object_box.rotation_base, dtype=np.float64).reshape(3, 3)
    dims = np.asarray(object_box.size_xyz, dtype=np.float64).reshape(3)
    best_axis = None
    best_score = -1.0
    for idx in range(3):
        axis_xy = rot[:2, idx]
        proj_norm = float(np.linalg.norm(axis_xy))
        score = float(dims[idx] * proj_norm)
        if score > best_score and proj_norm > 1e-6:
            best_score = score
            best_axis = np.array([float(axis_xy[0]), float(axis_xy[1]), 0.0], dtype=np.float64)
    long_axis = _normalize(best_axis, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    if float(long_axis[0]) > 0.0:
        long_axis = -long_axis
    return long_axis.astype(np.float64, copy=False)


def compute_default_pick_pose(object_box: OrientedBox) -> FramePose:
    z_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    x_axis = _default_pick_long_axis(object_box)
    y_axis = _normalize(np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    x_axis = _normalize(np.cross(y_axis, z_axis), fallback=x_axis)
    return FramePose(
        origin_base=object_box.top_center_base.astype(np.float32, copy=True),
        rotation_base=np.stack([x_axis, y_axis, z_axis], axis=1).astype(np.float32, copy=False),
    )


def compute_pick_pose_from_surface_point(object_box: OrientedBox, point_base: np.ndarray) -> FramePose:
    point = np.asarray(point_base, dtype=np.float64).reshape(3)
    local = _world_to_local(point.reshape(1, 3), object_box.center_base, object_box.rotation_base).reshape(3)
    half = 0.5 * object_box.size_xyz.astype(np.float64)
    local[0] = float(np.clip(local[0], -half[0], half[0]))
    local[1] = float(np.clip(local[1], -half[1], half[1]))
    local[2] = float(half[2])
    surface_point = _local_to_world(local.reshape(1, 3), object_box.center_base, object_box.rotation_base).reshape(3)
    pick_pose = compute_default_pick_pose(object_box)
    pick_pose.origin_base = surface_point.astype(np.float32, copy=False)
    return pick_pose


def _infer_horizontal_frame(points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = _as_points(points_base)
    if pts.shape[0] < 4:
        return np.zeros((3,), dtype=np.float32), np.eye(3, dtype=np.float32)
    xy = pts[:, :2].astype(np.float64, copy=False)
    center_seed = np.median(xy, axis=0)
    centered = xy - center_seed.reshape((1, 2))
    cov = centered.T @ centered / float(max(1, xy.shape[0]))
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    main_axis_xy = eigvecs[:, order[0]]
    pca_yaw = math.atan2(float(main_axis_xy[1]), float(main_axis_xy[0]))

    def _rect_metrics(yaw_rad: float) -> tuple[float, float, float, np.ndarray]:
        ex2 = np.array([math.cos(float(yaw_rad)), math.sin(float(yaw_rad))], dtype=np.float64)
        ey2 = np.array([-ex2[1], ex2[0]], dtype=np.float64)
        proj_x = xy @ ex2
        proj_y = xy @ ey2
        x0, x1 = np.quantile(proj_x, [0.04, 0.96])
        y0, y1 = np.quantile(proj_y, [0.04, 0.96])
        length = float(x1 - x0)
        width = float(y1 - y0)
        area = float(max(length, 1e-6) * max(width, 1e-6))
        return area, length, width, ex2

    best_yaw = float(_wrap_to_pi(pca_yaw))
    best_area, best_length, best_width, best_ex2 = _rect_metrics(best_yaw)
    search_steps = max(24, int(DEFAULT_FRAME_YAW_SEARCH_STEPS))
    for idx in range(search_steps):
        yaw = math.pi * float(idx) / float(search_steps)
        area, length, width, ex2 = _rect_metrics(yaw)
        better = area < best_area * 0.995
        tie = abs(area - best_area) <= 0.005 * max(best_area, 1e-6) and max(length, width) > max(best_length, best_width)
        if better or tie:
            best_yaw = float(yaw)
            best_area = float(area)
            best_length = float(length)
            best_width = float(width)
            best_ex2 = ex2

    if best_width > best_length:
        best_yaw = float(_wrap_to_pi(best_yaw + 0.5 * math.pi))
        _best_area, best_length, best_width, best_ex2 = _rect_metrics(best_yaw)

    ex = _normalize(np.array([float(best_ex2[0]), float(best_ex2[1]), 0.0], dtype=np.float64))
    ez = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ey = _normalize(np.cross(ez, ex), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float64))
    ex = _normalize(np.cross(ey, ez), fallback=ex)
    rot = np.stack([ex, ey, ez], axis=1).astype(np.float32, copy=False)
    return np.array([float(center_seed[0]), float(center_seed[1]), float(np.median(pts[:, 2]))], dtype=np.float32), rot


def _estimate_box_opening(
    local_centered: np.ndarray,
    size_xyz: np.ndarray,
    rotation_base: np.ndarray,
    center_base: np.ndarray,
    camera_origin_base: np.ndarray | None = None,
) -> tuple[dict[str, float], float, float, str | None, np.ndarray | None, dict[str, Any]]:
    half = 0.5 * np.asarray(size_xyz, dtype=np.float64).reshape(3)
    band = np.maximum(
        DEFAULT_FACE_BAND_RATIO * np.asarray(size_xyz, dtype=np.float64).reshape(3),
        np.array([0.010, 0.010, 0.010], dtype=np.float64),
    )

    def _face_density(axis: int, sign: int) -> float:
        coord = local_centered[:, axis]
        face_coord = float(sign) * float(half[axis] - band[axis])
        if sign > 0:
            return float(np.mean(coord >= face_coord))
        return float(np.mean(coord <= face_coord))

    densities = {
        "+x": _face_density(0, 1),
        "-x": _face_density(0, -1),
        "+y": _face_density(1, 1),
        "-y": _face_density(1, -1),
        "+z": _face_density(2, 1),
        "-z": _face_density(2, -1),
    }
    top_band = local_centered[:, 2] >= (half[2] - band[2])
    bottom_band = local_centered[:, 2] <= (-half[2] + band[2])
    inner_xy = (
        (np.abs(local_centered[:, 0]) <= max(half[0] - band[0], 0.0))
        & (np.abs(local_centered[:, 1]) <= max(half[1] - band[1], 0.0))
    )
    top_count = max(1, int(np.count_nonzero(top_band)))
    bottom_count = max(1, int(np.count_nonzero(bottom_band)))
    top_inner_fill = float(np.count_nonzero(top_band & inner_xy)) / float(top_count)
    bottom_inner_fill = float(np.count_nonzero(bottom_band & inner_xy)) / float(bottom_count)
    face_visibility = {face: 1.0 for face in densities.keys()}
    if camera_origin_base is not None:
        ctr = np.asarray(center_base, dtype=np.float64).reshape(3)
        rot = np.asarray(rotation_base, dtype=np.float64).reshape(3, 3)
        camera = np.asarray(camera_origin_base, dtype=np.float64).reshape(3)
        face_defs = (
            ("+x", rot[:, 0], half[0]),
            ("-x", -rot[:, 0], half[0]),
            ("+y", rot[:, 1], half[1]),
            ("-y", -rot[:, 1], half[1]),
            ("+z", rot[:, 2], half[2]),
            ("-z", -rot[:, 2], half[2]),
        )
        for face_name, outward_normal, face_offset in face_defs:
            face_center = ctr + outward_normal * float(face_offset)
            to_camera = camera - face_center
            to_camera_norm = float(np.linalg.norm(to_camera))
            if not np.isfinite(to_camera_norm) or to_camera_norm <= 1e-8:
                continue
            face_visibility[face_name] = float(
                max(0.0, np.dot(outward_normal, to_camera / to_camera_norm))
            )
    side_avg = float(np.mean([densities["+x"], densities["-x"], densities["+y"], densities["-y"]]))
    top_face_open_score = float(np.clip(1.0 - densities["+z"] / max(side_avg, 1e-4), 0.0, 1.0))
    top_inner_ref = max(bottom_inner_fill, side_avg, 1e-4)
    top_inner_open_score = float(np.clip(1.0 - top_inner_fill / top_inner_ref, 0.0, 1.0))
    top_open_score_raw = float(max(top_face_open_score, top_inner_open_score))
    top_visible = float(face_visibility.get("+z", 1.0)) >= float(DEFAULT_FACE_VISIBLE_MIN_COS)
    top_open_score = float(top_open_score_raw if top_visible else 0.0)

    horizontal_pairs = [
        ("+x", "-x", np.asarray(rotation_base, dtype=np.float64)[:, 0]),
        ("-x", "+x", -np.asarray(rotation_base, dtype=np.float64)[:, 0]),
        ("+y", "-y", np.asarray(rotation_base, dtype=np.float64)[:, 1]),
        ("-y", "+y", -np.asarray(rotation_base, dtype=np.float64)[:, 1]),
    ]
    opening_face: str | None = None
    opening_normal: np.ndarray | None = None
    front_open_score = 0.0
    for face_name, opposite_face, outward_normal in horizontal_pairs:
        if float(face_visibility.get(face_name, 1.0)) < float(DEFAULT_FACE_VISIBLE_MIN_COS):
            continue
        face_density = float(densities[face_name])
        opp_density = float(densities[opposite_face])
        if opp_density <= 1e-4:
            continue
        score = float(np.clip(1.0 - face_density / opp_density, 0.0, 1.0))
        if score > front_open_score:
            front_open_score = score
            opening_face = face_name
            opening_normal = outward_normal.astype(np.float64, copy=False)
    return (
        densities,
        top_open_score,
        front_open_score,
        opening_face,
        opening_normal,
        {
            "face_visibility": face_visibility,
            "box_top_inner_fill": float(top_inner_fill),
            "box_bottom_inner_fill": float(bottom_inner_fill),
            "box_top_face_open_score": float(top_face_open_score),
            "box_top_inner_open_score": float(top_inner_open_score),
            "box_top_open_score_raw": float(top_open_score_raw),
        },
    )


def _estimate_cylinder_opening(
    local_centered: np.ndarray,
    radius: float,
    height: float,
    center_base: np.ndarray | None = None,
    rotation_base: np.ndarray | None = None,
    camera_origin_base: np.ndarray | None = None,
) -> tuple[dict[str, float], float, float, str | None, np.ndarray | None, dict[str, float]]:
    pts = np.asarray(local_centered, dtype=np.float64)
    radial = np.linalg.norm(pts[:, :2], axis=1)
    half_h = 0.5 * float(height)
    band_r = max(0.006, DEFAULT_FACE_BAND_RATIO * float(radius))
    band_z = max(0.010, DEFAULT_FACE_BAND_RATIO * float(height))
    top_band = pts[:, 2] >= (half_h - band_z)
    bottom_band = pts[:, 2] <= (-half_h + band_z)
    shell_band = np.abs(radial - float(radius)) <= band_r
    inner_radius = max(0.0, float(radius) - 1.5 * band_r)
    inner_band = radial <= inner_radius
    top_count = max(1, int(np.count_nonzero(top_band)))
    bottom_count = max(1, int(np.count_nonzero(bottom_band)))
    top_inner_fill = float(np.count_nonzero(top_band & inner_band)) / float(top_count)
    bottom_inner_fill = float(np.count_nonzero(bottom_band & inner_band)) / float(bottom_count)
    shell_score = float(np.mean(shell_band)) if shell_band.size > 0 else 0.0
    top_open_ref = max(bottom_inner_fill, shell_score, 1e-4)
    top_open_score_raw = float(np.clip(1.0 - top_inner_fill / top_open_ref, 0.0, 1.0))
    top_visibility = 1.0
    if camera_origin_base is not None and center_base is not None and rotation_base is not None:
        ctr = np.asarray(center_base, dtype=np.float64).reshape(3)
        rot = np.asarray(rotation_base, dtype=np.float64).reshape(3, 3)
        camera = np.asarray(camera_origin_base, dtype=np.float64).reshape(3)
        top_center = ctr + rot[:, 2] * half_h
        to_camera = camera - top_center
        to_camera_norm = float(np.linalg.norm(to_camera))
        if np.isfinite(to_camera_norm) and to_camera_norm > 1e-8:
            top_visibility = float(max(0.0, np.dot(rot[:, 2], to_camera / to_camera_norm)))
    top_open_score = float(top_open_score_raw if top_visibility >= float(DEFAULT_FACE_VISIBLE_MIN_COS) else 0.0)
    densities = {
        "+x": float(shell_score),
        "-x": float(shell_score),
        "+y": float(shell_score),
        "-y": float(shell_score),
        "+z": float(top_inner_fill),
        "-z": float(bottom_inner_fill),
    }
    return (
        densities,
        top_open_score,
        0.0,
        None,
        None,
        {
            "cylinder_shell_score": float(shell_score),
            "cylinder_top_inner_fill": float(top_inner_fill),
            "cylinder_bottom_inner_fill": float(bottom_inner_fill),
            "face_visibility": {
                "+x": float(shell_score),
                "-x": float(shell_score),
                "+y": float(shell_score),
                "-y": float(shell_score),
                "+z": float(top_visibility),
                "-z": 0.0,
            },
        },
    )


def _select_target_kind(
    primitive_shape: str,
    size_xyz: np.ndarray,
    kind_hint: str,
    top_open_score: float,
    front_open_score: float,
    face_densities: dict[str, float],
) -> str:
    kind = str(kind_hint or DEFAULT_TARGET_KIND).strip().lower()
    if kind not in SUPPORTED_TARGET_KINDS:
        kind = DEFAULT_TARGET_KIND
    if kind != DEFAULT_TARGET_KIND:
        return str(kind)

    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    xy_max = float(max(size[0], size[1]))
    is_thin_surface = bool(
        primitive_shape == "plane"
        or size[2] <= 0.05
        or size[2] <= 0.28 * max(xy_max, 1e-4)
    )
    if is_thin_surface:
        return "support_surface"
    if primitive_shape == "cylinder" and top_open_score >= 0.40:
        return "top_open_box"
    if primitive_shape == "box" and front_open_score >= 0.42 and size[2] >= 0.10:
        return "front_open_cabinet"
    if top_open_score >= 0.40 and float(face_densities.get("-z", 0.0)) >= 0.02:
        return "top_open_box"
    if float(face_densities.get("+z", 0.0)) >= 0.04:
        return "closed_box"
    return "support_surface"


def _clone_target_profile(profile: TargetProfile) -> TargetProfile:
    return TargetProfile(
        kind=str(profile.kind),
        requested_kind=str(profile.requested_kind),
        primitive_shape=str(profile.primitive_shape),
        center_base=np.asarray(profile.center_base, dtype=np.float32).copy(),
        rotation_base=np.asarray(profile.rotation_base, dtype=np.float32).copy(),
        size_xyz=np.asarray(profile.size_xyz, dtype=np.float32).copy(),
        primitive_radius=None if profile.primitive_radius is None else float(profile.primitive_radius),
        top_z=float(profile.top_z),
        bottom_z=float(profile.bottom_z),
        floor_z=float(profile.floor_z),
        opening_face=profile.opening_face,
        opening_normal_base=None
        if profile.opening_normal_base is None
        else np.asarray(profile.opening_normal_base, dtype=np.float32).copy(),
        top_open_score=float(profile.top_open_score),
        front_open_score=float(profile.front_open_score),
        face_densities={str(key): float(val) for key, val in profile.face_densities.items()},
        points_base=np.asarray(profile.points_base, dtype=np.float32).copy(),
        debug=dict(profile.debug),
    )


def fit_target_profile(
    points_base: np.ndarray,
    kind_hint: str = DEFAULT_TARGET_KIND,
    camera_origin_base: np.ndarray | None = None,
) -> TargetProfile:
    pts = _as_points(points_base)
    if pts.shape[0] < 24:
        raise RuntimeError("Target point cloud is too small for placement planning.")

    seed_center, rot = _infer_horizontal_frame(pts)
    local = _world_to_local(pts, seed_center, rot)
    q_lo = np.quantile(local, 0.02, axis=0)
    q_hi = np.quantile(local, 0.98, axis=0)
    center_local = 0.5 * (q_lo + q_hi)
    size_xyz = np.maximum(q_hi - q_lo, 1e-4)
    center_world = _local_to_world(center_local.reshape(1, 3), seed_center, rot).reshape(3)
    local_centered = _world_to_local(pts, center_world, rot)

    size_xy = np.asarray(size_xyz[:2], dtype=np.float64)
    height = float(size_xyz[2])
    aspect_ratio_xy = float(min(size_xy) / max(max(size_xy), 1e-6))
    plane_like = bool(height <= max(0.04, 0.18 * max(float(size_xy[0]), float(size_xy[1]))))
    mid_band_half = max(0.020, 0.18 * max(height, 1e-4))
    mid_mask = np.abs(local_centered[:, 2]) <= mid_band_half
    mid_points = local_centered[mid_mask] if int(np.count_nonzero(mid_mask)) >= 24 else local_centered
    radial_mid = np.linalg.norm(mid_points[:, :2].astype(np.float64, copy=False), axis=1)
    radius = float(np.quantile(radial_mid, 0.90)) if radial_mid.size > 0 else 0.5 * float(np.mean(size_xy))
    radial_spread_ratio = 1.0
    if radial_mid.size > 0:
        q10_r, q50_r, q90_r = np.quantile(radial_mid, [0.10, 0.50, 0.90])
        radial_spread_ratio = float((q90_r - q10_r) / max(q50_r, 1e-6))
    band_box = np.maximum(
        DEFAULT_FACE_BAND_RATIO * np.maximum(size_xyz, 1e-4),
        np.array([0.010, 0.010, 0.010], dtype=np.float64),
    )
    half_box = 0.5 * np.asarray(size_xyz, dtype=np.float64).reshape(3)
    box_side_score = float(
        np.mean(
            (np.abs(np.abs(mid_points[:, 0]) - half_box[0]) <= band_box[0])
            | (np.abs(np.abs(mid_points[:, 1]) - half_box[1]) <= band_box[1])
        )
    ) if mid_points.shape[0] > 0 else 0.0
    cyl_band_r = max(0.006, DEFAULT_FACE_BAND_RATIO * max(radius, 1e-4))
    cyl_shell_score = float(np.mean(np.abs(radial_mid - radius) <= cyl_band_r)) if radial_mid.size > 0 else 0.0
    primitive_shape = "box"
    primitive_radius: float | None = None
    fit_debug: dict[str, Any] = {
        "aspect_ratio_xy": float(aspect_ratio_xy),
        "radial_spread_ratio": float(radial_spread_ratio),
        "box_side_score": float(box_side_score),
        "cylinder_shell_score": float(cyl_shell_score),
    }
    if plane_like:
        primitive_shape = "plane"
    elif aspect_ratio_xy >= 0.80 and radial_spread_ratio <= 0.16 and cyl_shell_score >= 0.30:
        primitive_shape = "cylinder"
        primitive_radius = float(max(radius, 0.5 * min(size_xy)))
        size_xyz[0] = float(max(size_xyz[0], 2.0 * primitive_radius))
        size_xyz[1] = float(max(size_xyz[1], 2.0 * primitive_radius))

    if primitive_shape == "cylinder":
        (
            densities,
            top_open_score,
            front_open_score,
            opening_face,
            opening_normal,
            cylinder_debug,
        ) = _estimate_cylinder_opening(
            local_centered.astype(np.float64, copy=False),
            float(primitive_radius or 0.5),
            height,
            center_base=center_world.astype(np.float64, copy=False),
            rotation_base=rot.astype(np.float64, copy=False),
            camera_origin_base=None
            if camera_origin_base is None
            else np.asarray(camera_origin_base, dtype=np.float64).reshape(3),
        )
        fit_debug.update(cylinder_debug)
    else:
        (
            densities,
            top_open_score,
            front_open_score,
            opening_face,
            opening_normal,
            box_debug,
        ) = _estimate_box_opening(
            local_centered.astype(np.float64, copy=False),
            size_xyz.astype(np.float64, copy=False),
            rot.astype(np.float64, copy=False),
            center_world.astype(np.float64, copy=False),
            camera_origin_base=None
            if camera_origin_base is None
            else np.asarray(camera_origin_base, dtype=np.float64).reshape(3),
        )
        fit_debug.update(box_debug)
        if primitive_shape == "plane":
            top_open_score = 0.0
            front_open_score = 0.0
            opening_face = None
            opening_normal = None

    half = 0.5 * size_xyz
    top_z = float(center_world[2] + half[2])
    bottom_z = float(center_world[2] - half[2])
    floor_z = float(_safe_quantile(pts[:, 2], 0.05, bottom_z))

    kind = _select_target_kind(
        primitive_shape=primitive_shape,
        size_xyz=size_xyz,
        kind_hint=kind_hint,
        top_open_score=top_open_score,
        front_open_score=front_open_score,
        face_densities=densities,
    )

    return TargetProfile(
        kind=str(kind),
        requested_kind=str(kind_hint),
        primitive_shape=str(primitive_shape),
        center_base=center_world.astype(np.float32, copy=False),
        rotation_base=rot.astype(np.float32, copy=False),
        size_xyz=size_xyz.astype(np.float32, copy=False),
        primitive_radius=None if primitive_radius is None else float(primitive_radius),
        top_z=float(top_z),
        bottom_z=float(bottom_z),
        floor_z=float(floor_z),
        opening_face=opening_face,
        opening_normal_base=None if opening_normal is None else opening_normal.astype(np.float32, copy=False),
        top_open_score=float(top_open_score),
        front_open_score=float(front_open_score),
        face_densities={key: float(val) for key, val in densities.items()},
        points_base=pts.astype(np.float32, copy=False),
        debug={
            "center_seed_base": [float(v) for v in seed_center.tolist()],
            "center_local": [float(v) for v in center_local.tolist()],
            "quantile_lo": [float(v) for v in q_lo.tolist()],
            "quantile_hi": [float(v) for v in q_hi.tolist()],
            "primitive_shape": str(primitive_shape),
            "primitive_radius": None if primitive_radius is None else float(primitive_radius),
            **fit_debug,
        },
    )


def _derive_object_to_gripper_local(object_box: OrientedBox, pick_pose_base: FramePose) -> tuple[np.ndarray, np.ndarray]:
    rel_origin = _world_to_local(
        pick_pose_base.origin_base.reshape(1, 3),
        object_box.center_base,
        object_box.rotation_base,
    ).reshape(3)
    rel_rotation = object_box.rotation_base.T @ pick_pose_base.rotation_base
    return rel_origin.astype(np.float32, copy=False), rel_rotation.astype(np.float32, copy=False)


def _compose_gripper_pose(
    object_box_place: OrientedBox,
    object_to_gripper_origin_local: np.ndarray,
    object_to_gripper_rotation_local: np.ndarray,
) -> FramePose:
    origin_world = _local_to_world(
        np.asarray(object_to_gripper_origin_local, dtype=np.float64).reshape(1, 3),
        object_box_place.center_base,
        object_box_place.rotation_base,
    ).reshape(3)
    rotation_world = object_box_place.rotation_base @ np.asarray(object_to_gripper_rotation_local, dtype=np.float64).reshape(3, 3)
    return FramePose(
        origin_base=origin_world.astype(np.float32, copy=False),
        rotation_base=rotation_world.astype(np.float32, copy=False),
    )


def _gripper_orientation_hard_constraint_notes(
    gripper_rotation_base: np.ndarray,
    config: PlannerConfig,
    allow_gripper_x_up: bool = False,
) -> list[str]:
    rot = np.asarray(gripper_rotation_base, dtype=np.float64).reshape(3, 3)
    gripper_x = rot[:, 0]
    gripper_z = rot[:, 2]
    notes: list[str] = []
    if float(gripper_z[0]) < float(config.gripper_forward_min_base_x):
        notes.append("gripper +Z points into the base -X hemisphere")
    if (not allow_gripper_x_up) and float(gripper_x[2]) > float(config.gripper_x_max_base_z):
        notes.append("gripper +X points into the base +Z hemisphere")
    return notes


def _candidate_yaws(profile: TargetProfile, planner_input: PlannerInput, config: PlannerConfig) -> list[float]:
    base = [
        profile.yaw_rad,
        profile.yaw_rad + math.radians(30.0),
        profile.yaw_rad - math.radians(30.0),
        profile.yaw_rad + 0.5 * math.pi,
        planner_input.object_box_current.yaw_rad,
        planner_input.object_box_current.yaw_rad + math.radians(30.0),
        planner_input.object_box_current.yaw_rad - math.radians(30.0),
        planner_input.object_box_current.yaw_rad + 0.5 * math.pi,
    ]
    if profile.opening_normal_base is not None:
        opening_yaw = math.atan2(
            float(profile.opening_normal_base[1]),
            float(profile.opening_normal_base[0]),
        )
        base.extend(
            [
                opening_yaw,
                opening_yaw + math.radians(20.0),
                opening_yaw - math.radians(20.0),
                opening_yaw + 0.5 * math.pi,
            ]
        )
    sweep_count = max(4, int(config.candidate_yaw_sweep_count))
    base.extend(
        [
            -math.pi + (2.0 * math.pi * float(idx) / float(sweep_count))
            for idx in range(sweep_count)
        ]
    )
    return _unique_angles(base, tol_rad=math.radians(2.0))


def _candidate_base_rotations(object_box: OrientedBox, profile: TargetProfile) -> list[tuple[str, np.ndarray]]:
    dims = np.asarray(object_box.size_xyz, dtype=np.float64).reshape(3)
    longest_axis = int(np.argmax(dims))
    bases: list[tuple[str, np.ndarray]] = [
        ("flat_z_up", np.eye(3, dtype=np.float64)),
        (
            "stand_x_up",
            np.array(
                [
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        ),
        (
            "stand_y_up",
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float64,
            ),
        ),
    ]
    if longest_axis == 2:
        bases = [bases[0], bases[1], bases[2]]
    elif longest_axis == 0:
        bases = [bases[0], bases[1], bases[2]]
    elif longest_axis == 1:
        bases = [bases[0], bases[2], bases[1]]
    if profile.kind in {"top_open_box", "front_open_cabinet"} or profile.primitive_shape == "cylinder":
        return bases
    return bases[:1]


def _candidate_object_rotations(
    profile: TargetProfile,
    planner_input: PlannerInput,
    object_to_gripper_rotation_local: np.ndarray,
    config: PlannerConfig,
) -> list[tuple[str, np.ndarray]]:
    out: list[tuple[str, np.ndarray]] = []
    rejected: list[tuple[str, np.ndarray]] = []
    object_to_gripper_rot = np.asarray(object_to_gripper_rotation_local, dtype=np.float64).reshape(3, 3)
    for yaw in _candidate_yaws(profile, planner_input, config):
        yaw_rot = _yaw_to_rotation(yaw)
        for base_name, base_rot in _candidate_base_rotations(planner_input.object_box_current, profile):
            rotation = yaw_rot @ np.asarray(base_rot, dtype=np.float64)
            gripper_rotation = rotation @ object_to_gripper_rot
            note = f"{base_name}@{math.degrees(yaw):.1f}deg"
            duplicate = any(
                np.max(np.abs(rotation - prev_rot)) <= 1e-4
                for _, prev_rot in out
            ) or any(
                np.max(np.abs(rotation - prev_rot)) <= 1e-4
                for _, prev_rot in rejected
            )
            if duplicate:
                continue
            if _gripper_orientation_hard_constraint_notes(gripper_rotation, config):
                rejected.append((note, rotation.astype(np.float64, copy=False)))
                continue
            out.append((note, rotation.astype(np.float64, copy=False)))
    return out if out else rejected


def _candidate_relaxed_top_open_rotations(
    profile: TargetProfile,
    planner_input: PlannerInput,
    object_to_gripper_rotation_local: np.ndarray,
    config: PlannerConfig,
) -> list[tuple[str, np.ndarray]]:
    if str(profile.kind) != "top_open_box":
        return []
    longest_axis, _, slender_ratio = _slender_metrics(planner_input.object_box_current.size_xyz)
    if slender_ratio < DEFAULT_RELAXED_TOP_OPEN_MIN_SLENDER_RATIO:
        return []
    out: list[tuple[str, np.ndarray]] = []
    rejected: list[tuple[str, np.ndarray]] = []
    object_to_gripper_rot = np.asarray(object_to_gripper_rotation_local, dtype=np.float64).reshape(3, 3)
    for yaw in _candidate_yaws(profile, planner_input, config):
        yaw_rot = _yaw_to_rotation(yaw)
        yaw_axes = (
            ("tilt_x", yaw_rot[:, 0]),
            ("tilt_y", yaw_rot[:, 1]),
        )
        for base_name, base_rot in _candidate_base_rotations(planner_input.object_box_current, profile):
            base_rotation = yaw_rot @ np.asarray(base_rot, dtype=np.float64)
            if abs(float(base_rotation[2, longest_axis])) < 0.55:
                continue
            for axis_name, world_axis in yaw_axes:
                for angle_deg in DEFAULT_RELAXED_TOP_OPEN_TILT_ANGLES_DEG:
                    angle_rad = math.radians(float(angle_deg))
                    for angle_sign in (-1.0, 1.0):
                        rotation = _rotation_about_axis(world_axis, angle_sign * angle_rad) @ base_rotation
                        gripper_rotation = rotation @ object_to_gripper_rot
                        note = f"{base_name}+{axis_name}{angle_sign * float(angle_deg):+.0f}deg@{math.degrees(yaw):.1f}deg"
                        duplicate = any(
                            np.max(np.abs(rotation - prev_rot)) <= 1e-4
                            for _, prev_rot in out
                        ) or any(
                            np.max(np.abs(rotation - prev_rot)) <= 1e-4
                            for _, prev_rot in rejected
                        )
                        if duplicate:
                            continue
                        if _gripper_orientation_hard_constraint_notes(gripper_rotation, config):
                            rejected.append((note, rotation.astype(np.float64, copy=False)))
                            continue
                        out.append((note, rotation.astype(np.float64, copy=False)))
    return out if out else rejected


def _vertical_half_extent(size_xyz: np.ndarray, rotation_base: np.ndarray) -> float:
    size = np.asarray(size_xyz, dtype=np.float64).reshape(3)
    rot = np.asarray(rotation_base, dtype=np.float64).reshape(3, 3)
    return float(0.5 * np.sum(np.abs(rot[2, :]) * size))


def _score_arm_clearance(
    origin_base: np.ndarray,
    arm_axis_base: np.ndarray,
    scene_points_base: np.ndarray,
    max_distance_m: float,
    radius_m: float,
) -> tuple[float, float]:
    pts = _as_points(scene_points_base)
    if pts.shape[0] == 0:
        return 1.0, float(max_distance_m)
    origin = np.asarray(origin_base, dtype=np.float64).reshape(-1)
    if origin.shape[0] < 3 or not np.all(np.isfinite(origin[:3])):
        return 0.0, 0.0
    axis = _normalize(np.asarray(arm_axis_base, dtype=np.float64).reshape(3), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float64))
    diff = pts.astype(np.float64, copy=False) - origin[:3].reshape(1, 3)
    axial = diff @ axis.reshape(3, 1)
    axial = axial.reshape(-1)
    radial_vec = diff - axial.reshape((-1, 1)) * axis.reshape((1, 3))
    radial = np.linalg.norm(radial_vec, axis=1)
    keep = np.isfinite(axial) & np.isfinite(radial) & (axial > 0.0) & (axial <= float(max_distance_m)) & (radial <= float(radius_m))
    if not np.any(keep):
        return 1.0, float(max_distance_m)
    nearest = float(np.min(axial[keep]))
    score = float(np.clip(nearest / max(float(max_distance_m), 1e-6), 0.0, 1.0))
    return score, nearest


def _candidate_grasp_rotation_variants_local(object_to_gripper_rotation_local: np.ndarray) -> list[tuple[str, np.ndarray]]:
    base_rot = np.asarray(object_to_gripper_rotation_local, dtype=np.float64).reshape(3, 3)
    variants: list[tuple[str, np.ndarray]] = [("as-picked", base_rot.astype(np.float64, copy=False))]
    flip_z = base_rot @ _rotation_about_axis(np.array([0.0, 0.0, 1.0], dtype=np.float64), math.pi)
    if not any(np.max(np.abs(flip_z - prev_rot)) <= 1e-4 for _, prev_rot in variants):
        variants.append(("wrist-yaw-180", flip_z.astype(np.float64, copy=False)))
    return variants


def _candidate_invalid_penalty(candidate: PlacementCandidate) -> tuple[int, int, float]:
    severity = {
        "gripper +Z points into the base -X hemisphere": 200,
        "gripper +X points into the base +Z hemisphere": 160,
        "placed object OBB is outside camera frame": 190,
        "place grasp origin is outside camera frame": 210,
        "scene collision points intersect placed object OBB": 140,
        "object exceeds cabinet inner footprint": 80,
        "object exceeds top-open container inner footprint": 80,
        "object exceeds top-open container inner radius": 80,
        "object footprint exceeds support surface bounds": 70,
        "object footprint exceeds cylindrical support bounds": 70,
        "object is below cabinet floor": 60,
        "object is taller than the top-open container usable depth": 50,
        "object is taller than cabinet opening": 50,
    }
    penalty = 0
    counted = 0
    for note in candidate.notes:
        reason = str(note)
        if reason.startswith("orientation=") or reason.startswith("grasp_variant="):
            continue
        if (
            reason.startswith("place ")
            or reason.startswith("fallback ")
            or reason.startswith("cabinet fallback ")
            or reason.startswith("attempt ")
            or reason.startswith("allow partial support overhang")
        ):
            continue
        penalty += int(severity.get(reason, 10))
        counted += 1
    return penalty, counted, -float(candidate.score)


def _candidate_grasp_variant_label(candidate: PlacementCandidate) -> str:
    for note in candidate.notes:
        reason = str(note)
        if reason.startswith("grasp_variant="):
            return str(reason.split("=", 1)[1]).strip()
    return ""


def _candidate_pose_key(candidate: PlacementCandidate) -> tuple[str, tuple[float, ...], tuple[float, ...]]:
    center = tuple(
        float(v)
        for v in np.round(
            np.asarray(candidate.object_box_place.center_base, dtype=np.float64).reshape(3),
            4,
        )
    )
    rotation = tuple(
        float(v)
        for v in np.round(
            np.asarray(candidate.object_box_place.rotation_base, dtype=np.float64).reshape(-1),
            4,
        )
    )
    return str(candidate.rule_id), center, rotation


def _filter_redundant_grasp_variant_candidates(
    valid_candidates: list[PlacementCandidate],
) -> list[PlacementCandidate]:
    if len(valid_candidates) <= 1:
        return list(valid_candidates)
    grouped: dict[tuple[str, tuple[float, ...], tuple[float, ...]], list[PlacementCandidate]] = {}
    for candidate in valid_candidates:
        grouped.setdefault(_candidate_pose_key(candidate), []).append(candidate)
    filtered: list[PlacementCandidate] = []
    for group in grouped.values():
        as_picked = [candidate for candidate in group if _candidate_grasp_variant_label(candidate) == "as-picked"]
        if as_picked:
            filtered.extend(as_picked)
        else:
            filtered.extend(group)
    return filtered


def _score_candidate(
    rule_id: str,
    object_box_place: OrientedBox,
    gripper_pose_place: FramePose,
    profile: TargetProfile,
    planner_input: PlannerInput,
    config: PlannerConfig,
    allow_gripper_x_up: bool = False,
) -> tuple[bool, dict[str, float], list[str]]:
    notes: list[str] = []
    corners = object_box_place.corners_world().astype(np.float64)
    target_local = profile.world_to_local(corners).astype(np.float64)
    obj_height = float(object_box_place.size_xyz[2])
    half_target = 0.5 * profile.size_xyz.astype(np.float64)
    edge_clear = float(min(float(config.edge_clearance_m), 0.12 * float(np.min(profile.size_xyz[:2]))))
    wall_clear = float(min(float(config.container_wall_clearance_m), 0.08 * float(np.min(profile.size_xyz[:2]))))
    rim_clear = float(min(float(config.container_rim_clearance_m), 0.12 * float(profile.size_xyz[2])))
    support_radius = float(profile.primitive_radius or 0.5 * min(float(profile.size_xyz[0]), float(profile.size_xyz[1])))
    support_coverage_score = 0.5

    valid = True
    if rule_id in {"support_surface", "closed_box"}:
        bottom_samples_world = _sample_object_box_bottom_face_world(object_box_place).astype(np.float64, copy=False)
        bottom_samples_local = profile.world_to_local(bottom_samples_world).astype(np.float64, copy=False)
        raw_inside = _support_surface_local_inside_mask(profile, bottom_samples_local, margin_m=0.0)
        safe_inside = _support_surface_local_inside_mask(profile, bottom_samples_local, margin_m=edge_clear)
        support_coverage_score = float(np.mean(raw_inside)) if raw_inside.size > 0 else 0.0
        support_safe_coverage = float(np.mean(safe_inside)) if safe_inside.size > 0 else support_coverage_score
        center_local = profile.world_to_local(
            np.asarray(object_box_place.center_base, dtype=np.float32).reshape(1, 3)
        ).astype(np.float64, copy=False)
        center_inside = bool(_support_surface_local_inside_mask(profile, center_local, margin_m=0.0)[0])
        min_support_coverage = _support_surface_min_coverage(object_box_place, profile)
        if not center_inside or support_coverage_score < min_support_coverage:
            valid = False
            if str(profile.primitive_shape) == "cylinder":
                notes.append("object footprint exceeds cylindrical support bounds")
            else:
                notes.append("object footprint exceeds support surface bounds")
        elif support_safe_coverage < 0.999:
            notes.append("allow partial support overhang")
    elif rule_id == "top_open_box":
        max_allowed_top_z = float(profile.top_z - rim_clear)
        volume_samples_world = _sample_object_box_volume_world(object_box_place).astype(np.float64, copy=False)
        volume_samples_local = profile.world_to_local(volume_samples_world).astype(np.float64, copy=False)
        below_rim = volume_samples_local[:, 2] <= (half_target[2] - rim_clear + 1e-6)
        if not np.any(below_rim):
            below_rim = np.ones((volume_samples_local.shape[0],), dtype=bool)
        interior_samples = volume_samples_local[below_rim]
        if profile.primitive_shape == "cylinder":
            radial = np.linalg.norm(interior_samples[:, :2], axis=1)
            if np.any(radial > max(1e-4, support_radius - wall_clear)):
                valid = False
                notes.append("object exceeds top-open container inner radius")
        else:
            if np.any(np.abs(interior_samples[:, 0]) > (half_target[0] - wall_clear)) or np.any(
                np.abs(interior_samples[:, 1]) > (half_target[1] - wall_clear)
            ):
                valid = False
                notes.append("object exceeds top-open container inner footprint")
        if float(np.max(corners[:, 2])) > max_allowed_top_z:
            obj_dims_sorted = np.sort(np.asarray(object_box_place.size_xyz, dtype=np.float64))
            slender_ratio = float(obj_dims_sorted[2] / max(obj_dims_sorted[1], 1e-4))
            insert_depth = float(max_allowed_top_z - np.min(corners[:, 2]))
            min_insert_depth = float(max(0.03, 1.6 * obj_dims_sorted[0]))
            if slender_ratio >= 3.0 and insert_depth >= min_insert_depth:
                notes.append("allow tall slender object to protrude above container rim")
            else:
                valid = False
                notes.append("object is taller than the top-open container usable depth")
    elif rule_id == "front_open_cabinet":
        if np.any(np.abs(target_local[:, 0]) > (half_target[0] - wall_clear)) or np.any(
            np.abs(target_local[:, 1]) > (half_target[1] - wall_clear)
        ):
            valid = False
            notes.append("object exceeds cabinet inner footprint")
        if float(np.min(corners[:, 2])) < float(profile.floor_z + config.support_clearance_m):
            valid = False
            notes.append("object is below cabinet floor")
        if float(np.max(corners[:, 2])) > float(profile.top_z - rim_clear):
            valid = False
            notes.append("object is taller than cabinet opening")

    scene_local = _world_to_local(
        planner_input.scene_collision_points_base,
        object_box_place.center_base,
        object_box_place.rotation_base,
    )
    if scene_local.shape[0] > 0:
        half = 0.5 * object_box_place.size_xyz.astype(np.float64).reshape(1, 3)
        inside = np.all(np.abs(scene_local) <= (half + float(config.scene_collision_margin_m)), axis=1)
        if np.any(inside):
            valid = False
            notes.append("scene collision points intersect placed object OBB")

    gripper_x = gripper_pose_place.x_axis_base.astype(np.float64, copy=False)
    gripper_z = gripper_pose_place.z_axis_base.astype(np.float64, copy=False)
    hard_notes = _gripper_orientation_hard_constraint_notes(
        gripper_pose_place.rotation_base,
        config,
        allow_gripper_x_up=allow_gripper_x_up,
    )
    if hard_notes:
        valid = False
        notes.extend(hard_notes)
    x_down_score = float(np.clip(-gripper_x[2], 0.0, 1.0))
    arm_axis = -gripper_z
    arm_clearance_score, arm_clearance_distance = _score_arm_clearance(
        gripper_pose_place.origin_base,
        arm_axis,
        planner_input.scene_collision_points_base,
        max_distance_m=float(config.arm_clearance_max_distance_m),
        radius_m=float(config.arm_clearance_radius_m),
    )

    opening_alignment_score = 0.5
    if profile.opening_normal_base is not None:
        opening_alignment_score = float(
            np.clip(
                0.5 * (1.0 + float(np.dot(arm_axis, profile.opening_normal_base.astype(np.float64, copy=False)))),
                0.0,
                1.0,
            )
        )

    height_bias = float(np.clip((float(profile.top_z) + 0.01 - float(object_box_place.center_base[2])) / max(obj_height, 1e-4), -1.0, 1.0))
    height_score = float(np.clip(0.5 + 0.5 * height_bias, 0.0, 1.0))
    pick_to_place_rotation_delta_rad = 0.0
    pick_to_place_rotation_score = 0.5
    if planner_input.start_gripper_axes_base is not None:
        pick_to_place_rotation_delta_rad = _rotation_geodesic_angle(
            planner_input.start_gripper_axes_base,
            gripper_pose_place.rotation_base,
        )
        pick_to_place_rotation_score = float(
            np.clip(1.0 - pick_to_place_rotation_delta_rad / math.pi, 0.0, 1.0)
        )
    pick_to_place_translation_distance_m = 0.0
    pick_to_place_translation_score = 0.5
    if planner_input.start_object_center_base is not None:
        pick_to_place_translation_distance_m = float(
            np.linalg.norm(
                np.asarray(object_box_place.center_base, dtype=np.float64).reshape(3)
                - np.asarray(planner_input.start_object_center_base, dtype=np.float64).reshape(3)
            )
        )
        distance_ref = max(
            float(config.arm_clearance_max_distance_m),
            2.0 * float(np.linalg.norm(np.asarray(planner_input.object_box_current.size_xyz, dtype=np.float64).reshape(3))),
            1e-4,
        )
        pick_to_place_translation_score = float(
            np.clip(1.0 - pick_to_place_translation_distance_m / distance_ref, 0.0, 1.0)
        )

    camera_visibility_debug: dict[str, Any] | None = None
    if planner_input.camera_visibility_constraint is not None:
        camera_visibility_debug = _evaluate_candidate_camera_visibility(
            object_box_place,
            gripper_pose_place,
            planner_input.camera_visibility_constraint,
        )
        if bool(camera_visibility_debug.get("available", False)):
            if (
                planner_input.camera_visibility_constraint.require_full_object_in_frame
                and not bool(camera_visibility_debug.get("object", {}).get("all_in_frame", False))
            ):
                valid = False
                notes.append("placed object OBB is outside camera frame")
            if (
                planner_input.camera_visibility_constraint.require_grasp_origin_in_frame
                and not bool(camera_visibility_debug.get("grasp", {}).get("all_in_frame", False))
            ):
                valid = False
                notes.append("place grasp origin is outside camera frame")

    score_terms = {
        "x_down": float(x_down_score),
        "arm_clearance": float(arm_clearance_score),
        "opening_alignment": float(opening_alignment_score),
        "height_bias": float(height_score),
        "support_coverage": float(support_coverage_score),
        "pick_to_place_rotation": float(pick_to_place_rotation_score),
        "pick_to_place_translation": float(pick_to_place_translation_score),
        "arm_clearance_distance_m": float(arm_clearance_distance),
        "pick_to_place_rotation_delta_rad": float(pick_to_place_rotation_delta_rad),
        "pick_to_place_translation_distance_m": float(pick_to_place_translation_distance_m),
    }
    if camera_visibility_debug is not None and bool(camera_visibility_debug.get("available", False)):
        score_terms["camera_object_all_in_frame"] = float(bool(camera_visibility_debug.get("object", {}).get("all_in_frame", False)))
        score_terms["camera_grasp_all_in_frame"] = float(bool(camera_visibility_debug.get("grasp", {}).get("all_in_frame", False)))
    return valid, score_terms, notes


def _finalize_candidate(
    rule_id: str,
    object_box_place: OrientedBox,
    object_to_gripper_origin_local: np.ndarray,
    object_to_gripper_rotation_local: np.ndarray,
    profile: TargetProfile,
    planner_input: PlannerInput,
    config: PlannerConfig,
    notes: list[str] | None = None,
    allow_gripper_x_up: bool = False,
) -> PlacementCandidate:
    candidate_notes = list(notes or [])
    gripper_pose_place = _compose_gripper_pose(
        object_box_place,
        object_to_gripper_origin_local,
        object_to_gripper_rotation_local,
    )
    valid, score_terms, validation_notes = _score_candidate(
        rule_id,
        object_box_place,
        gripper_pose_place,
        profile,
        planner_input,
        config,
        allow_gripper_x_up=allow_gripper_x_up,
    )
    candidate_notes.extend(validation_notes)
    total_score = (
        2.7 * float(score_terms.get("arm_clearance", 0.0))
        + float(config.x_down_score_weight) * float(score_terms.get("x_down", 0.0))
        + 1.5 * float(score_terms.get("opening_alignment", 0.0))
        + 0.4 * float(score_terms.get("height_bias", 0.0))
        + float(config.support_coverage_score_weight) * float(score_terms.get("support_coverage", 0.0))
        + float(config.pick_to_place_rotation_score_weight) * float(score_terms.get("pick_to_place_rotation", 0.0))
        + float(config.pick_to_place_translation_score_weight) * float(score_terms.get("pick_to_place_translation", 0.0))
    )
    return PlacementCandidate(
        rule_id=str(rule_id),
        object_box_place=object_box_place,
        gripper_pose_place=gripper_pose_place,
        score=float(total_score),
        score_terms=score_terms,
        notes=candidate_notes,
        valid=bool(valid),
    )


class SupportSurfaceRule:
    rule_id = "support_surface"

    def match_score(self, profile: TargetProfile, planner_input: PlannerInput) -> float:
        if profile.kind == self.rule_id:
            return 1.0
        if profile.kind == "closed_box":
            return 0.85
        return 0.25

    def generate_candidates(
        self,
        profile: TargetProfile,
        planner_input: PlannerInput,
        object_to_gripper_origin_local: np.ndarray,
        object_to_gripper_rotation_local: np.ndarray,
        config: PlannerConfig,
    ) -> list[PlacementCandidate]:
        obj = planner_input.object_box_current
        candidates: list[PlacementCandidate] = []
        center_offsets_local = _support_surface_center_offsets_local(
            profile,
            planner_input.preferred_center_base,
        )
        for rotation_note, rotation in _candidate_object_rotations(
            profile,
            planner_input,
            object_to_gripper_rotation_local,
            config,
        ):
            vertical_half = _vertical_half_extent(obj.size_xyz, rotation)
            center_local_z = float(0.5 * float(profile.size_xyz[2]) + vertical_half + config.support_clearance_m)
            for center_note, center_xy_local in center_offsets_local:
                center_local = np.array(
                    [
                        float(center_xy_local[0]),
                        float(center_xy_local[1]),
                        center_local_z,
                    ],
                    dtype=np.float64,
                )
                center_world = profile.local_to_world(center_local.reshape(1, 3)).reshape(3)
                placed = OrientedBox(center_base=center_world, rotation_base=rotation, size_xyz=obj.size_xyz)
                candidates.append(
                    _finalize_candidate(
                        rule_id=self.rule_id,
                        object_box_place=placed,
                        object_to_gripper_origin_local=object_to_gripper_origin_local,
                        object_to_gripper_rotation_local=object_to_gripper_rotation_local,
                        profile=profile,
                        planner_input=planner_input,
                        config=config,
                        notes=[f"place on support {center_note}", f"orientation={rotation_note}"],
                    )
                )
        return candidates


class ClosedBoxRule(SupportSurfaceRule):
    rule_id = "closed_box"

    def match_score(self, profile: TargetProfile, planner_input: PlannerInput) -> float:
        return 1.0 if profile.kind == self.rule_id else 0.0


class TopOpenBoxRule:
    rule_id = "top_open_box"

    def match_score(self, profile: TargetProfile, planner_input: PlannerInput) -> float:
        if profile.kind == self.rule_id:
            return 1.0
        return 0.0

    def generate_candidates(
        self,
        profile: TargetProfile,
        planner_input: PlannerInput,
        object_to_gripper_origin_local: np.ndarray,
        object_to_gripper_rotation_local: np.ndarray,
        config: PlannerConfig,
    ) -> list[PlacementCandidate]:
        obj = planner_input.object_box_current
        candidates: list[PlacementCandidate] = []
        strict_rotations = _candidate_object_rotations(
            profile,
            planner_input,
            object_to_gripper_rotation_local,
            config,
        )
        for rotation_note, rotation in strict_rotations:
            vertical_half = _vertical_half_extent(obj.size_xyz, rotation)
            center_inside = np.array(
                [
                    float(profile.center_base[0]),
                    float(profile.center_base[1]),
                    float(profile.floor_z + vertical_half + config.support_clearance_m),
                ],
                dtype=np.float32,
            )
            placed_inside = OrientedBox(center_base=center_inside, rotation_base=rotation, size_xyz=obj.size_xyz)
            candidates.append(
                _finalize_candidate(
                    rule_id=self.rule_id,
                    object_box_place=placed_inside,
                    object_to_gripper_origin_local=object_to_gripper_origin_local,
                    object_to_gripper_rotation_local=object_to_gripper_rotation_local,
                    profile=profile,
                    planner_input=planner_input,
                    config=config,
                    notes=[
                        "attempt inside top-open container",
                        f"orientation={rotation_note}",
                    ],
                )
            )
            center_top = np.array(
                [
                    float(profile.center_base[0]),
                    float(profile.center_base[1]),
                    float(profile.top_z + vertical_half + config.support_clearance_m),
                ],
                dtype=np.float32,
            )
            placed_top = OrientedBox(center_base=center_top, rotation_base=rotation, size_xyz=obj.size_xyz)
            candidates.append(
                _finalize_candidate(
                    rule_id="support_surface",
                    object_box_place=placed_top,
                    object_to_gripper_origin_local=object_to_gripper_origin_local,
                    object_to_gripper_rotation_local=object_to_gripper_rotation_local,
                    profile=profile,
                    planner_input=planner_input,
                    config=config,
                    notes=["fallback place on top surface", f"orientation={rotation_note}"],
                )
            )
        for rotation_note, rotation in _candidate_relaxed_top_open_rotations(
            profile,
            planner_input,
            object_to_gripper_rotation_local,
            config,
        ):
            vertical_half = _vertical_half_extent(obj.size_xyz, rotation)
            center_inside = np.array(
                [
                    float(profile.center_base[0]),
                    float(profile.center_base[1]),
                    float(profile.floor_z + vertical_half + config.support_clearance_m),
                ],
                dtype=np.float32,
            )
            placed_inside = OrientedBox(center_base=center_inside, rotation_base=rotation, size_xyz=obj.size_xyz)
            candidates.append(
                _finalize_candidate(
                    rule_id=self.rule_id,
                    object_box_place=placed_inside,
                    object_to_gripper_origin_local=object_to_gripper_origin_local,
                    object_to_gripper_rotation_local=object_to_gripper_rotation_local,
                    profile=profile,
                    planner_input=planner_input,
                    config=config,
                    notes=[
                        "attempt relaxed tilted insertion into top-open container",
                        f"orientation={rotation_note}",
                    ],
                )
            )
        return candidates


class FrontOpenCabinetRule:
    rule_id = "front_open_cabinet"

    def match_score(self, profile: TargetProfile, planner_input: PlannerInput) -> float:
        return 1.0 if profile.kind == self.rule_id else 0.0

    def generate_candidates(
        self,
        profile: TargetProfile,
        planner_input: PlannerInput,
        object_to_gripper_origin_local: np.ndarray,
        object_to_gripper_rotation_local: np.ndarray,
        config: PlannerConfig,
    ) -> list[PlacementCandidate]:
        obj = planner_input.object_box_current
        candidates: list[PlacementCandidate] = []
        for rotation_note, rotation in _candidate_object_rotations(
            profile,
            planner_input,
            object_to_gripper_rotation_local,
            config,
        ):
            vertical_half = _vertical_half_extent(obj.size_xyz, rotation)
            center_local = np.array(
                [
                    0.0,
                    0.0,
                    float(profile.floor_z - profile.center_base[2] + vertical_half + config.support_clearance_m),
                ],
                dtype=np.float64,
            )
            center_world = profile.local_to_world(center_local.reshape(1, 3)).reshape(3)
            placed = OrientedBox(center_base=center_world, rotation_base=rotation, size_xyz=obj.size_xyz)
            candidates.append(
                _finalize_candidate(
                    rule_id=self.rule_id,
                    object_box_place=placed,
                    object_to_gripper_origin_local=object_to_gripper_origin_local,
                    object_to_gripper_rotation_local=object_to_gripper_rotation_local,
                    profile=profile,
                    planner_input=planner_input,
                    config=config,
                    notes=["place inside front-open cabinet", f"orientation={rotation_note}"],
                )
            )
            center_top = np.array(
                [
                    float(profile.center_base[0]),
                    float(profile.center_base[1]),
                    float(profile.top_z + vertical_half + config.support_clearance_m),
                ],
                dtype=np.float32,
            )
            placed_top = OrientedBox(center_base=center_top, rotation_base=rotation, size_xyz=obj.size_xyz)
            candidates.append(
                _finalize_candidate(
                    rule_id="support_surface",
                    object_box_place=placed_top,
                    object_to_gripper_origin_local=object_to_gripper_origin_local,
                    object_to_gripper_rotation_local=object_to_gripper_rotation_local,
                    profile=profile,
                    planner_input=planner_input,
                    config=config,
                    notes=["cabinet fallback place on top surface", f"orientation={rotation_note}"],
                )
            )
        return candidates


DEFAULT_RULES: tuple[PlacementRule, ...] = (
    FrontOpenCabinetRule(),
    TopOpenBoxRule(),
    ClosedBoxRule(),
    SupportSurfaceRule(),
)


class PlaceDofPlanner:
    def __init__(self, config: PlannerConfig | None = None, rules: tuple[PlacementRule, ...] | None = None) -> None:
        self.config = config if config is not None else PlannerConfig()
        self.rules = rules if rules is not None else DEFAULT_RULES

    def build_target_profile(self, planner_input: PlannerInput) -> TargetProfile:
        if planner_input.target_profile_override is not None:
            return _clone_target_profile(planner_input.target_profile_override)
        points = planner_input.target_points_base
        if points.shape[0] < 24 and planner_input.table_points_base.shape[0] >= 24:
            points = planner_input.table_points_base
        profile = fit_target_profile(points, planner_input.target_kind_hint)
        if planner_input.preferred_center_base is not None:
            preferred = np.asarray(planner_input.preferred_center_base, dtype=np.float32).reshape(3)
            profile.center_base[:2] = preferred[:2]
            if profile.kind == "support_surface":
                profile.center_base[2] = float(preferred[2])
        return profile

    def plan(self, planner_input: PlannerInput) -> PlacementPlanResult:
        profile = self.build_target_profile(planner_input)
        rel_origin, rel_rotation = _derive_object_to_gripper_local(
            planner_input.object_box_current,
            planner_input.pick_pose_base,
        )
        grasp_rotation_variants = _candidate_grasp_rotation_variants_local(rel_rotation)

        ranked_rules = sorted(
            self.rules,
            key=lambda rule: float(rule.match_score(profile, planner_input)),
            reverse=True,
        )
        all_candidates: list[PlacementCandidate] = []
        best_rule_id: str | None = None
        for rule in ranked_rules:
            rule_match_score = float(rule.match_score(profile, planner_input))
            if rule_match_score <= 0.0:
                continue
            for grasp_variant_note, grasp_variant_rot in grasp_rotation_variants:
                generated = rule.generate_candidates(
                    profile=profile,
                    planner_input=planner_input,
                    object_to_gripper_origin_local=rel_origin,
                    object_to_gripper_rotation_local=grasp_variant_rot,
                    config=self.config,
                )
                for candidate in generated:
                    candidate.score += 0.2 * rule_match_score
                    grasp_variant_preference = 1.0 if grasp_variant_note == "as-picked" else 0.0
                    candidate.score += float(self.config.as_picked_grasp_score_weight) * grasp_variant_preference
                    candidate.score_terms["grasp_variant_preference"] = float(grasp_variant_preference)
                    candidate.notes.append(f"grasp_variant={grasp_variant_note}")
                all_candidates.extend(generated)
                if generated and best_rule_id is None:
                    best_rule_id = str(rule.rule_id)

        valid_candidates = [candidate for candidate in all_candidates if candidate.valid]
        valid_candidates = _filter_redundant_grasp_variant_candidates(valid_candidates)
        valid_by_match_score: dict[float, list[PlacementCandidate]] = {}
        for candidate in valid_candidates:
            candidate_match_score = float(
                next(
                    (
                        rule.match_score(profile, planner_input)
                        for rule in ranked_rules
                        if str(rule.rule_id) == str(candidate.rule_id)
                    ),
                    0.0,
                )
            )
            valid_by_match_score.setdefault(candidate_match_score, []).append(candidate)
        valid_candidates.sort(key=lambda candidate: float(candidate.score), reverse=True)
        invalid_reason_counts: dict[str, int] = {}
        for candidate in all_candidates:
            if candidate.valid:
                continue
            if candidate.notes:
                for note in candidate.notes:
                    reason = str(note)
                    if reason.startswith("orientation="):
                        continue
                    if (
                        reason.startswith("place ")
                        or reason.startswith("fallback ")
                        or reason.startswith("cabinet fallback ")
                        or reason.startswith("attempt ")
                    ):
                        continue
                    invalid_reason_counts[reason] = int(invalid_reason_counts.get(reason, 0)) + 1
            else:
                invalid_reason_counts["candidate invalid without explicit note"] = int(
                    invalid_reason_counts.get("candidate invalid without explicit note", 0) + 1
                )
        best_candidate = None
        best_effort_selected = False
        if valid_by_match_score:
            best_match_score = max(valid_by_match_score.keys())
            preferred_pool_raw = list(valid_by_match_score.get(best_match_score, []))
            if str(profile.kind) in {"top_open_box", "front_open_cabinet", "closed_box"}:
                native_pool = [
                    candidate
                    for candidate in preferred_pool_raw
                    if str(candidate.rule_id) == str(profile.kind)
                ]
                if native_pool:
                    preferred_pool_raw = native_pool
            preferred_pool = sorted(
                preferred_pool_raw,
                key=lambda candidate: float(candidate.score),
                reverse=True,
            )
            if preferred_pool:
                best_candidate = preferred_pool[0]
        if best_candidate is None and valid_candidates:
            best_candidate = valid_candidates[0]
        if best_candidate is None and all_candidates and planner_input.allow_best_effort:
            fallback_pool = sorted(
                all_candidates,
                key=_candidate_invalid_penalty,
            )
            if fallback_pool:
                best_candidate = fallback_pool[0]
                best_effort_selected = True
        success = best_candidate is not None
        if success and best_candidate is not None:
            best_rule_id = str(best_candidate.rule_id)
        return PlacementPlanResult(
            success=bool(success),
            best_candidate=best_candidate,
            all_candidates=sorted(all_candidates, key=lambda candidate: float(candidate.score), reverse=True),
            target_profile=profile,
            selected_rule_id=best_rule_id,
            debug={
                "candidate_count": int(len(all_candidates)),
                "valid_candidate_count": int(len(valid_candidates)),
                "target_kind": str(profile.kind),
                "target_requested_kind": str(profile.requested_kind),
                "target_primitive_shape": str(profile.primitive_shape),
                "target_primitive_radius": None if profile.primitive_radius is None else float(profile.primitive_radius),
                "target_top_open_score": float(profile.top_open_score),
                "target_front_open_score": float(profile.front_open_score),
                "target_face_densities": {key: float(val) for key, val in profile.face_densities.items()},
                "invalid_reason_counts": dict(sorted(invalid_reason_counts.items(), key=lambda item: item[1], reverse=True)),
                "object_to_gripper_origin_local": [float(v) for v in rel_origin.tolist()],
                "object_to_gripper_rotation_local": np.asarray(rel_rotation, dtype=np.float32).tolist(),
                "grasp_rotation_variant_count": int(len(grasp_rotation_variants)),
                "allow_best_effort": bool(planner_input.allow_best_effort),
                "best_effort_selected": bool(best_effort_selected),
            },
        )


@dataclass
class OBBSpec:
    size_xyz: np.ndarray

    def __post_init__(self) -> None:
        self.size_xyz = np.maximum(np.asarray(self.size_xyz, dtype=np.float32).reshape(3), 1e-4)


@dataclass
class RelativePose:
    origin_local: np.ndarray
    axes_local: np.ndarray

    def __post_init__(self) -> None:
        self.origin_local = np.asarray(self.origin_local, dtype=np.float32).reshape(3)
        self.axes_local = FramePose(
            origin_base=np.zeros((3,), dtype=np.float32),
            rotation_base=np.asarray(self.axes_local, dtype=np.float32).reshape(3, 3),
        ).rotation_base


@dataclass
class PlannerRequest:
    obb: OBBSpec
    grasp_pose_local: RelativePose
    target_points_base: np.ndarray
    scene_points_base: np.ndarray
    table_points_base: np.ndarray = field(default_factory=_empty_points)
    preferred_center_base: np.ndarray | None = None
    target_kind_hint: str = DEFAULT_TARGET_KIND
    target_profile_override: TargetProfile | None = None
    start_object_center_base: np.ndarray | None = None
    start_gripper_axes_base: np.ndarray | None = None
    camera_visibility_constraint: CameraVisibilityConstraint | None = None
    allow_best_effort: bool = True
    debug_hints: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_points_base = _as_points(self.target_points_base)
        self.scene_points_base = _as_points(self.scene_points_base)
        self.table_points_base = _as_points(self.table_points_base)
        if self.preferred_center_base is not None:
            center = np.asarray(self.preferred_center_base, dtype=np.float32).reshape(-1)
            if center.shape[0] >= 3 and np.all(np.isfinite(center[:3])):
                self.preferred_center_base = center[:3].astype(np.float32, copy=False)
            else:
                self.preferred_center_base = None
        if self.start_object_center_base is not None:
            center = np.asarray(self.start_object_center_base, dtype=np.float32).reshape(-1)
            if center.shape[0] >= 3 and np.all(np.isfinite(center[:3])):
                self.start_object_center_base = center[:3].astype(np.float32, copy=False)
            else:
                self.start_object_center_base = None
        if self.start_gripper_axes_base is not None:
            axes = np.asarray(self.start_gripper_axes_base, dtype=np.float32)
            if axes.shape == (3, 3) and np.all(np.isfinite(axes)):
                self.start_gripper_axes_base = FramePose(
                    origin_base=np.zeros((3,), dtype=np.float32),
                    rotation_base=axes,
                ).rotation_base
            else:
                self.start_gripper_axes_base = None
        self.allow_best_effort = bool(self.allow_best_effort)


@dataclass
class TargetAnalysis:
    kind: str
    face_coverage: dict[str, float]
    debug: dict[str, Any]


@dataclass
class PlannerBestCandidate:
    object_center_base: np.ndarray
    object_axes_base: np.ndarray
    object_rpy_rad: np.ndarray
    gripper_origin_base: np.ndarray
    gripper_axes_base: np.ndarray
    rule_id: str
    target_kind: str
    score: float
    score_breakdown: dict[str, float]
    debug: dict[str, Any]


@dataclass
class PlannerResponse:
    success: bool
    best: PlannerBestCandidate | None
    matched_rules: list[str]
    target_analysis: TargetAnalysis
    debug: dict[str, Any]


def _rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sy = math.sqrt(float(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = math.atan2(float(-rot[1, 2]), float(rot[1, 1]))
        pitch = math.atan2(float(-rot[2, 0]), sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw], dtype=np.float32)


def make_default_grasp_pose_local(obb: OBBSpec) -> RelativePose:
    object_box = OrientedBox(
        center_base=np.zeros((3,), dtype=np.float32),
        rotation_base=np.eye(3, dtype=np.float32),
        size_xyz=np.asarray(obb.size_xyz, dtype=np.float32),
    )
    pose = compute_default_pick_pose(object_box)
    return RelativePose(origin_local=pose.origin_base, axes_local=pose.rotation_base)


def plan_placement(
    request: PlannerRequest,
    planner: PlaceDofPlanner | None = None,
) -> PlannerResponse:
    solver = planner if planner is not None else PlaceDofPlanner()
    dummy_object = OrientedBox(
        center_base=np.zeros((3,), dtype=np.float32),
        rotation_base=np.eye(3, dtype=np.float32),
        size_xyz=np.asarray(request.obb.size_xyz, dtype=np.float32),
    )
    pick_pose_base = FramePose(
        origin_base=np.asarray(request.grasp_pose_local.origin_local, dtype=np.float32),
        rotation_base=np.asarray(request.grasp_pose_local.axes_local, dtype=np.float32),
    )
    result = solver.plan(
        PlannerInput(
            object_box_current=dummy_object,
            pick_pose_base=pick_pose_base,
            target_points_base=request.target_points_base,
            scene_collision_points_base=request.scene_points_base,
            table_points_base=request.table_points_base,
            preferred_center_base=request.preferred_center_base,
            target_kind_hint=request.target_kind_hint,
            target_profile_override=request.target_profile_override,
            start_object_center_base=request.start_object_center_base,
            start_gripper_axes_base=request.start_gripper_axes_base,
            camera_visibility_constraint=request.camera_visibility_constraint,
            allow_best_effort=request.allow_best_effort,
            debug=dict(request.debug_hints),
        )
    )
    best_candidate = None
    if result.best_candidate is not None:
        best_candidate = PlannerBestCandidate(
            object_center_base=np.asarray(result.best_candidate.object_box_place.center_base, dtype=np.float32),
            object_axes_base=np.asarray(result.best_candidate.object_box_place.rotation_base, dtype=np.float32),
            object_rpy_rad=_rotation_matrix_to_rpy(result.best_candidate.object_box_place.rotation_base),
            gripper_origin_base=np.asarray(result.best_candidate.gripper_pose_place.origin_base, dtype=np.float32),
            gripper_axes_base=np.asarray(result.best_candidate.gripper_pose_place.rotation_base, dtype=np.float32),
            rule_id=str(result.best_candidate.rule_id),
            target_kind=str(result.target_profile.kind),
            score=float(result.best_candidate.score),
            score_breakdown={str(key): float(val) for key, val in result.best_candidate.score_terms.items()},
            debug={
                "notes": list(result.best_candidate.notes),
                "valid": bool(result.best_candidate.valid),
            },
        )
    matched_rules: list[str] = []
    for candidate in result.all_candidates:
        rule_id = str(candidate.rule_id)
        if rule_id not in matched_rules:
            matched_rules.append(rule_id)
    return PlannerResponse(
        success=bool(result.success),
        best=best_candidate,
        matched_rules=matched_rules,
        target_analysis=TargetAnalysis(
            kind=str(result.target_profile.kind),
            face_coverage={str(key): float(val) for key, val in result.target_profile.face_densities.items()},
            debug={
                "primitive_shape": str(result.target_profile.primitive_shape),
                "primitive_radius": None
                if result.target_profile.primitive_radius is None
                else float(result.target_profile.primitive_radius),
                "top_open_score": float(result.target_profile.top_open_score),
                "front_open_score": float(result.target_profile.front_open_score),
                "opening_face": result.target_profile.opening_face,
                **dict(result.target_profile.debug),
            },
        ),
        debug=dict(result.debug),
    )
