#!/usr/bin/env python3
"""
Post-Grasp In-Hand Pose Re-Estimation (抓取后手中物体位姿重估计)

两种工作模式:
  模式 1 (world): 提供 cam_to_base → 在 base 系下配准 → 直接输出新的 base 系 OBB
  模式 2 (camera): 不提供 cam_to_base → 在相机系下配准 → 输出 T_slip (夹爪系滑动量)

输入:
  - 抓取前/后 手部双目图像 (左图 + 右图) + 内参 + baseline
  - 深度由 S2M2 云端接口生成
  - Mask 由 SAM3 自动分割

启动:
    cd GraspPipeline
    python main_pose_in_hand.py                      # 默认 http://0.0.0.0:7880
    python main_pose_in_hand.py --port 7881 --share
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

try:
    import gradio as gr
except ImportError:
    gr = None  # type: ignore[assignment]

try:
    import open3d as o3d
except ImportError:
    o3d = None  # type: ignore[assignment]

import logging
import requests

from main_pcd_fit_cam import (
    fit_camera_primitive_from_points,
    render_best_fit_overlay_image,
    serialize_fit_result,
    summarize_fit_result_lines,
)
from pcd_registration_core import (
    CameraIntrinsics,
    RegistrationConfig,
    copy_pcd,
    invert_transform,
    local_hypothesis_search,
    make_transform,
    pca_initial_transform,
    pca_initial_transform_candidates,
    preprocess_pcd,
    refine_with_colored_icp,
    refine_with_gicp,
    rgbd_to_pointcloud,
    transform_points,
    visibility_aware_score,
)

_log = logging.getLogger(__name__)

DEFAULT_SAM3_URL = os.getenv(
    "SAM3_SERVER_URL",
    os.getenv("PLACEDOF_SAM3_URL", "http://101.132.143.105:5081"),
).rstrip("/")
DEFAULT_S2M2_URL = os.getenv(
    "S2M2_API_URL",
    os.getenv("PLACEDOF_S2M2_API_URL", "http://101.132.143.105:5056/api/process"),
).rstrip("/")
QWEN_DASHSCOPE_API_KEY = os.getenv("QWEN_DASHSCOPE_API_KEY", "sk-4a47da58c2e64a53bc7b94d0892016be")
QWEN_GROUNDING_MODEL = os.getenv("QWEN_GROUNDING_MODEL", "qwen3-vl-30b-a3b-instruct")
_DASHSCOPE_MULTIMODAL_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


# ============================================================================
# Self-contained cloud service clients (S2M2 / SAM3 / Qwen)
# ============================================================================

def _call_s2m2_depth_api(
    left_img_path: str, right_img_path: str,
    fx: float, fy: float, cx: float, cy: float,
    baseline: float, server_url: str = DEFAULT_S2M2_URL,
) -> np.ndarray:
    """Call S2M2 depth estimation. Returns depth/xyz_map numpy array."""
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_str = " ".join(str(v) for row in K for v in row)
    left_f = open(left_img_path, "rb")
    right_f = open(right_img_path, "rb")
    try:
        resp = requests.post(
            server_url,
            files={"left_file": left_f, "right_file": right_f},
            data={"K": K_str, "baseline": str(np.float32(baseline))},
            timeout=600,
        )
    finally:
        left_f.close()
        right_f.close()
    if resp.status_code != 200:
        raise RuntimeError(f"S2M2 failed [{resp.status_code}]: {resp.text[:300]}")
    return np.load(io.BytesIO(resp.content))


def _call_sam3_segment_api(
    image_base64: str,
    box_prompt: list | None = None,
    server_url: str = DEFAULT_SAM3_URL,
) -> dict | None:
    """Call SAM3 segmentation with a bbox prompt. Returns JSON dict or None."""
    payload: dict = {"image": image_base64}
    if box_prompt:
        payload["box_prompt"] = box_prompt
    if len(payload) < 2:
        return None
    try:
        resp = requests.post(
            f"{server_url}/segment", json=payload,
            headers={"Content-Type": "application/json"}, timeout=60,
        )
        return resp.json() if resp.status_code == 200 else None
    except Exception:
        return None


def _call_qwen_grounding_api(
    image_b64: str, prompt: str,
    img_width: int, img_height: int,
) -> dict:
    """Call Qwen VLM grounding to detect object bbox from a text prompt."""
    find_prompt = (
        f"Your task is to find {prompt} in the image, which should be on the table or some platform.\n"
        "Note: \n"
        "1. If color or texture is provided, you should carefully check the item that matches the description."
        "2. Choose only ONE most likely item if you find multiple ones.\n"
        "3. Strictly following the <OUTPUT FORMAT> in JSON format.\n"
        "4. Do not consider the item that is far from the table or platform."
        "5. If you cannot find the item, return <finished:true>."
    )
    messages = [
        {"role": "system", "content": [{"text": "You are a helpful tracking assistant."}]},
        {"role": "user", "content": [
            {"text": "Perform a pick and place task in the given image. If the pick or place conditions are not met, return <finished:true>."},
            {"image": f"data:image/jpeg;base64,{image_b64}", "min_pixels": 640*480, "max_pixels": 1280*960},
            {"text": find_prompt},
        ]},
        {"role": "user", "content": [{"text": '<OUTPUT TEMPLATE>: {"bbox": [x1, y1, x2, y2]}'}]},
    ]
    try:
        resp = requests.post(
            _DASHSCOPE_MULTIMODAL_URL,
            headers={"Authorization": f"Bearer {QWEN_DASHSCOPE_API_KEY}", "Content-Type": "application/json"},
            json={"model": QWEN_GROUNDING_MODEL, "input": {"messages": messages}},
            timeout=120,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"DashScope [{resp.status_code}]: {resp.text[:300]}")
        choices = resp.json().get("output", {}).get("choices", [])
        if not choices:
            raise RuntimeError("DashScope returned no choices")
        msg_text = (choices[0].get("message", {}).get("content", [{}])[0].get("text", ""))
        for i, line in enumerate(msg_text.splitlines()):
            if line == "```json":
                msg_text = "\n".join(msg_text.splitlines()[i+1:]).split("```")[0]
                break
        if "true" in msg_text:
            return {"bbox": None, "qwen_status": "finished", "response": msg_text}
        msg_json, _ = json.JSONDecoder().raw_decode(msg_text.strip())
        x1, y1, x2, y2 = msg_json["bbox"]
        x1, y1 = int(x1/1000*img_width), int(y1/1000*img_height)
        x2, y2 = int(x2/1000*img_width), int(y2/1000*img_height)
        return {"bbox": [x1, y1, x2, y2], "qwen_status": "success", "response": msg_text}
    except Exception as e:
        _log.error("Qwen grounding failed ('%s'): %s", prompt, e)
        return {"bbox": None, "qwen_status": "error", "response": str(e)}

CUSTOM_CSS = """
.gradio-container { max-width: 1700px !important; margin: auto; }
#app-title { text-align: center; padding: 10px 0 2px; }
#app-title h1 { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
.summary-box textarea {
    font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 0.82rem !important; line-height: 1.5 !important;
}
"""


# ============================================================================
# Data classes
# ============================================================================

@dataclass
class OBBState:
    center_base: np.ndarray
    axes_base: np.ndarray
    size_xyz: np.ndarray

    def __post_init__(self) -> None:
        self.center_base = np.asarray(self.center_base, dtype=np.float32).reshape(3)
        self.axes_base = np.asarray(self.axes_base, dtype=np.float32).reshape(3, 3)
        self.size_xyz = np.asarray(self.size_xyz, dtype=np.float32).reshape(3)

    def copy(self) -> OBBState:
        return OBBState(self.center_base.copy(), self.axes_base.copy(), self.size_xyz.copy())

    def corners_world(self) -> np.ndarray:
        half = 0.5 * self.size_xyz.astype(np.float64)
        signs = np.array(
            [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
             [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]],
            dtype=np.float64,
        )
        local = signs * half
        return (local @ self.axes_base.astype(np.float64).T + self.center_base.astype(np.float64)).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "center_base": self.center_base.tolist(),
            "axes_base": self.axes_base.tolist(),
            "size_xyz": self.size_xyz.tolist(),
        }


@dataclass
class ReEstimationResult:
    success: bool
    mode: str  # "world" or "camera"
    T_registration: np.ndarray
    T_slip: np.ndarray
    old_obb: OBBState | None = None
    new_obb: OBBState | None = None
    debug: Dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0


# ============================================================================
# Utility functions
# ============================================================================

def _orthonormalize_axes(axes: np.ndarray) -> np.ndarray:
    a = np.asarray(axes, dtype=np.float64).reshape(3, 3)
    x = a[:, 0]
    x = x / max(np.linalg.norm(x), 1e-12)
    y = a[:, 1] - np.dot(a[:, 1], x) * x
    y = y / max(np.linalg.norm(y), 1e-12)
    z = np.cross(x, y)
    z = z / max(np.linalg.norm(z), 1e-12)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def apply_transform_to_obb(obb: OBBState, T: np.ndarray) -> OBBState:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    new_center = (R @ obb.center_base.astype(np.float64) + t).astype(np.float32)
    new_axes = _orthonormalize_axes((R @ obb.axes_base.astype(np.float64)).astype(np.float32))
    return OBBState(center_base=new_center, axes_base=new_axes, size_xyz=obb.size_xyz.copy())


def compute_T_slip(
    T_reg: np.ndarray,
    T_wrist_to_ee: np.ndarray | None = None,
) -> np.ndarray:
    """
    Convert camera-frame registration result to object slip in gripper frame.

    T_slip = T_wrist_to_ee @ T_reg @ T_wrist_to_ee^{-1}

    If T_wrist_to_ee is None (unknown hand-eye calibration), T_slip = T_reg.
    """
    T_reg = np.asarray(T_reg, dtype=np.float64).reshape(4, 4)
    if T_wrist_to_ee is None:
        return T_reg.copy()
    T_we = np.asarray(T_wrist_to_ee, dtype=np.float64).reshape(4, 4)
    return T_we @ T_reg @ np.linalg.inv(T_we)


def apply_slip_to_obb(
    old_obb: OBBState,
    T_slip: np.ndarray,
    G_at_pick: np.ndarray,
    G_current: np.ndarray,
) -> OBBState:
    """
    Apply T_slip + gripper motion to recover OBB in base frame.

    new_obb = G_current @ T_slip @ G_at_pick^{-1} @ old_obb
    """
    G_pick = np.asarray(G_at_pick, dtype=np.float64).reshape(4, 4)
    G_curr = np.asarray(G_current, dtype=np.float64).reshape(4, 4)
    T_full = G_curr @ np.asarray(T_slip, dtype=np.float64).reshape(4, 4) @ np.linalg.inv(G_pick)
    return apply_transform_to_obb(old_obb, T_full)


def _t_slip_frame_name(mode: str, T_wrist_to_ee: np.ndarray | None) -> str:
    if T_wrist_to_ee is not None:
        return "ee"
    if mode == "world":
        return "base"
    return "camera"


def _t_slip_axis_description(frame_name: str) -> str:
    if frame_name == "camera":
        return "camera axes: +x right, +y down, +z forward"
    if frame_name == "ee":
        return "ee/gripper axes: uses the axes defined by your T_wrist_to_ee"
    if frame_name == "base":
        return "base axes: uses the axes of the provided world/base frame"
    return f"{frame_name} axes"


def _t_slip_rotation_lines(
    T_slip: np.ndarray,
    frame_name: str,
    compact: bool = False,
) -> list[str]:
    lines: list[str] = []
    try:
        from scipy.spatial.transform import Rotation

        rot = Rotation.from_matrix(np.asarray(T_slip, dtype=np.float64)[:3, :3])
        euler = rot.as_euler("xyz", degrees=True)
        rotvec = rot.as_rotvec()
        angle_deg = float(np.linalg.norm(rotvec) * 180.0 / np.pi)
        if angle_deg > 1e-9:
            axis = rotvec / max(np.linalg.norm(rotvec), 1e-12)
        else:
            axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        if compact:
            lines.append(
                f"T_slip rot: Euler xyz extrinsic [{euler[0]:.1f}, {euler[1]:.1f}, {euler[2]:.1f}]deg"
            )
            lines.append(f"T_slip frame: {_t_slip_axis_description(frame_name)}")
            lines.append(f"Axis-angle: axis=[{axis[0]:.2f}, {axis[1]:.2f}, {axis[2]:.2f}], angle={angle_deg:.1f}deg")
        else:
            lines.append(f"  旋转(Euler xyz, extrinsic / fixed axes): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")
            lines.append(f"  坐标轴: {_t_slip_axis_description(frame_name)}")
            lines.append("  欧拉角转序: xyz, 即 Rx -> Ry -> Rz（等价矩阵写法 R = Rz @ Ry @ Rx）")
            lines.append(f"  轴角(axis-angle): axis=[{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}], angle={angle_deg:.2f} deg")
    except ImportError:
        pass
    return lines


def _rot_axis_deg(axis: str, angle_deg: float) -> np.ndarray:
    rad = np.deg2rad(float(angle_deg))
    c, s = float(np.cos(rad)), float(np.sin(rad))
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    raise ValueError(f"Unsupported axis: {axis}")


def _rotate_transform_about_src_centroid(
    T_init: np.ndarray,
    src_pts: np.ndarray,
    axis: str = "z",
    angle_deg: float = 180.0,
) -> np.ndarray:
    src_centroid = src_pts.mean(axis=0).astype(np.float64)
    pivot = (T_init[:3, :3] @ src_centroid + T_init[:3, 3]).astype(np.float64)
    dR = _rot_axis_deg(axis, angle_deg)
    rotated_pivot = dR @ pivot
    dT = make_transform(dR, pivot - rotated_pivot)
    return dT @ T_init


def _append_unique_init(
    items: list[tuple[str, np.ndarray]],
    name: str,
    T: np.ndarray,
    rot_tol: float = 1e-5,
    trans_tol: float = 1e-6,
) -> None:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    for _, existing in items:
        if (
            np.linalg.norm(existing[:3, :3] - T[:3, :3]) <= rot_tol
            and np.linalg.norm(existing[:3, 3] - T[:3, 3]) <= trans_tol
        ):
            return
    items.append((name, T))


def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(arr))
    if not np.isfinite(n) or n <= eps:
        return np.zeros(3, dtype=np.float64)
    return arr / n


def _shape_type_family(fit_type: str | None) -> str | None:
    if fit_type in {"cylinder", "frustum"}:
        return "elongated_solid"
    if fit_type in {"bowl", "plate"}:
        return "dish"
    return fit_type


def _shape_type_compatibility(src_type: str | None, tgt_type: str | None) -> float:
    if not src_type or not tgt_type:
        return 0.0
    if src_type == tgt_type:
        return 1.0
    src_family = _shape_type_family(src_type)
    tgt_family = _shape_type_family(tgt_type)
    if src_family == tgt_family == "elongated_solid":
        return 0.78
    if src_family == tgt_family == "dish":
        return 0.72
    return 0.0


def _shape_primitive_scale(fit_type: str | None, primitive: Any) -> float:
    if primitive is None or fit_type is None:
        return 0.0
    if fit_type == "cuboid":
        return float(np.max(np.asarray(primitive.size, dtype=np.float64)))
    if fit_type == "cylinder":
        return float(max(primitive.height, 2.0 * primitive.semi_major, 2.0 * primitive.semi_minor))
    if fit_type == "frustum":
        return float(max(
            primitive.height,
            2.0 * primitive.small_end_major,
            2.0 * primitive.large_end_major,
        ))
    if fit_type == "bowl":
        return float(max(2.0 * primitive.rim_major, primitive.depth))
    if fit_type == "plate":
        return float(max(2.0 * primitive.rim_major, primitive.depth))
    return 0.0


def _shape_fit_confidence(fit_result: Dict[str, Any] | None) -> float:
    if not fit_result:
        return 0.0
    fit_type = fit_result.get("best_type")
    primitive = fit_result.get("best_result")
    if fit_type is None or primitive is None:
        return 0.0

    scale = max(_shape_primitive_scale(fit_type, primitive), 1e-4)
    residual = float(getattr(primitive, "residual", 0.0))
    residual_score = float(np.exp(-((residual / max(0.10 * scale, 0.0035)) ** 2)))

    scores = fit_result.get("scores") or {}
    best_score = float(scores.get(fit_type, residual))
    others = [float(v) for k, v in scores.items() if k != fit_type]
    if others:
        second = min(others)
        margin = max(second - best_score, 0.0) / max(second, 1e-6)
        margin_score = float(np.clip(margin / 0.25, 0.0, 1.0))
    else:
        margin_score = 0.75
    return float(np.clip(0.65 * residual_score + 0.35 * margin_score, 0.0, 1.0))


def _shape_fit_variants(
    fit_result: Dict[str, Any] | None,
    rel_margin: float = 0.14,
    abs_margin: float = 2.5e-4,
    max_variants: int = 3,
) -> list[Dict[str, Any]]:
    if not fit_result:
        return []
    scores = fit_result.get("scores") or {}
    all_results = fit_result.get("all_results") or {}
    best_type = fit_result.get("best_type")
    best_result = fit_result.get("best_result")
    if best_type is None or best_result is None:
        return []

    if not scores:
        variant = dict(fit_result)
        variant["_variant_rank"] = 0
        variant["_variant_score"] = float(getattr(best_result, "residual", 0.0))
        variant["_variant_rel_gap"] = 0.0
        variant["_variant_base_type"] = best_type
        return [variant]

    ordered = sorted(scores.items(), key=lambda kv: kv[1])
    best_score = float(scores.get(best_type, ordered[0][1]))
    variants: list[Dict[str, Any]] = []
    for rank, (cand_type, score) in enumerate(ordered):
        cand_result = all_results.get(cand_type)
        if cand_result is None:
            if cand_type == best_type:
                cand_result = best_result
            else:
                continue
        rel_gap = max(float(score) - best_score, 0.0) / max(float(score), 1e-8)
        include = (
            rank == 0
            or float(score) <= best_score + abs_margin
            or rel_gap <= rel_margin
        )
        if not include:
            continue
        variant = dict(fit_result)
        variant["best_type"] = cand_type
        variant["best_result"] = cand_result
        variant["_variant_rank"] = rank
        variant["_variant_score"] = float(score)
        variant["_variant_rel_gap"] = float(rel_gap)
        variant["_variant_base_type"] = best_type
        variants.append(variant)
        if len(variants) >= max_variants:
            break

    if not variants:
        variant = dict(fit_result)
        variant["_variant_rank"] = 0
        variant["_variant_score"] = float(best_score)
        variant["_variant_rel_gap"] = 0.0
        variant["_variant_base_type"] = best_type
        variants.append(variant)
    return variants


def _select_effective_shape_pair(
    src_fit: Dict[str, Any] | None,
    tgt_fit: Dict[str, Any] | None,
    T_src_frame: np.ndarray,
    T_tgt_frame: np.ndarray,
) -> tuple[Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any] | None, Dict[str, Any]]:
    info: Dict[str, Any] = {
        "selection_mode": "inactive",
        "selected": False,
        "source_type": None,
        "target_type": None,
        "source_best_type": None,
        "target_best_type": None,
        "source_variant_rank": None,
        "target_variant_rank": None,
        "compatibility": 0.0,
        "confidence": 0.0,
        "scale_score": 0.0,
        "pair_score": 0.0,
    }
    if src_fit is None or tgt_fit is None:
        return src_fit, tgt_fit, None, None, info

    src_variants = _shape_fit_variants(src_fit)
    tgt_variants = _shape_fit_variants(tgt_fit)
    if not src_variants or not tgt_variants:
        return src_fit, tgt_fit, None, None, info

    primary_src = src_variants[0]
    primary_tgt = tgt_variants[0]
    primary_src_desc = _shape_fit_to_descriptor(primary_src, T_src_frame)
    primary_tgt_desc = _shape_fit_to_descriptor(primary_tgt, T_tgt_frame)
    primary_compat = _shape_type_compatibility(primary_src.get("best_type"), primary_tgt.get("best_type"))
    primary_conf = min(
        float(primary_src_desc.get("confidence", 0.0)) if primary_src_desc else 0.0,
        float(primary_tgt_desc.get("confidence", 0.0)) if primary_tgt_desc else 0.0,
    )
    if primary_compat > 0.0 and primary_conf >= 0.42 and primary_src_desc is not None and primary_tgt_desc is not None:
        info.update({
            "selection_mode": "primary_best",
            "selected": True,
            "source_type": primary_src.get("best_type"),
            "target_type": primary_tgt.get("best_type"),
            "source_best_type": src_fit.get("best_type"),
            "target_best_type": tgt_fit.get("best_type"),
            "source_variant_rank": primary_src.get("_variant_rank", 0),
            "target_variant_rank": primary_tgt.get("_variant_rank", 0),
            "compatibility": float(primary_compat),
            "confidence": float(primary_conf),
            "scale_score": 1.0,
            "pair_score": float(primary_compat * primary_conf),
        })
        return primary_src, primary_tgt, primary_src_desc, primary_tgt_desc, info

    best_pair: tuple[Any, ...] | None = None
    best_pair_score = -1.0
    for src_variant in src_variants:
        src_desc = _shape_fit_to_descriptor(src_variant, T_src_frame)
        if src_desc is None:
            continue
        for tgt_variant in tgt_variants:
            tgt_desc = _shape_fit_to_descriptor(tgt_variant, T_tgt_frame)
            if tgt_desc is None:
                continue
            compat = _shape_type_compatibility(src_variant.get("best_type"), tgt_variant.get("best_type"))
            conf = min(
                float(src_desc.get("confidence", 0.0)),
                float(tgt_desc.get("confidence", 0.0)),
            )
            if compat <= 0.0 or conf < 0.42:
                continue
            src_scale = float(max(src_desc.get("scale", 0.0), 1e-6))
            tgt_scale = float(max(tgt_desc.get("scale", 0.0), 1e-6))
            scale_rel = abs(src_scale - tgt_scale) / max(src_scale, tgt_scale, 1e-6)
            scale_score = float(np.exp(-((scale_rel / 0.35) ** 2)))
            rank_penalty = max(0.72, 1.0 - 0.08 * (
                float(src_variant.get("_variant_rank", 0)) + float(tgt_variant.get("_variant_rank", 0))
            ))
            family_bonus = 1.05 if src_desc.get("family") == tgt_desc.get("family") else 1.0
            pair_score = float(compat * conf * scale_score * rank_penalty * family_bonus)
            if pair_score > best_pair_score:
                best_pair_score = pair_score
                best_pair = (src_variant, tgt_variant, src_desc, tgt_desc, compat, conf, scale_score, pair_score)

    if best_pair is not None:
        src_variant, tgt_variant, src_desc, tgt_desc, compat, conf, scale_score, pair_score = best_pair
        info.update({
            "selection_mode": "ambiguous_fallback",
            "selected": True,
            "source_type": src_variant.get("best_type"),
            "target_type": tgt_variant.get("best_type"),
            "source_best_type": src_fit.get("best_type"),
            "target_best_type": tgt_fit.get("best_type"),
            "source_variant_rank": src_variant.get("_variant_rank", 0),
            "target_variant_rank": tgt_variant.get("_variant_rank", 0),
            "compatibility": float(compat),
            "confidence": float(conf),
            "scale_score": float(scale_score),
            "pair_score": float(pair_score),
        })
        return src_variant, tgt_variant, src_desc, tgt_desc, info

    return primary_src, primary_tgt, primary_src_desc, primary_tgt_desc, info


def _align_vectors_rotation(src_axis: np.ndarray, tgt_axis: np.ndarray) -> np.ndarray:
    src = _safe_normalize(src_axis)
    tgt = _safe_normalize(tgt_axis)
    if np.linalg.norm(src) < 1e-8 or np.linalg.norm(tgt) < 1e-8:
        return np.eye(3, dtype=np.float64)
    cross = np.cross(src, tgt)
    cross_norm = float(np.linalg.norm(cross))
    dot = float(np.clip(np.dot(src, tgt), -1.0, 1.0))
    if cross_norm <= 1e-8:
        if dot > 0.0:
            return np.eye(3, dtype=np.float64)
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(src[0]) > 0.9:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _safe_normalize(np.cross(src, helper))
        K = np.array([
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ], dtype=np.float64)
        return np.eye(3, dtype=np.float64) + 2.0 * (K @ K)

    axis = cross / cross_norm
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + np.sin(np.arccos(dot)) * K + (1.0 - dot) * (K @ K)


def _shape_transform_point(pt: np.ndarray, T_cam_to_frame: np.ndarray) -> np.ndarray:
    pt = np.asarray(pt, dtype=np.float64).reshape(3)
    T = np.asarray(T_cam_to_frame, dtype=np.float64).reshape(4, 4)
    return (T[:3, :3] @ pt + T[:3, 3]).astype(np.float64)


def _shape_transform_rotation(R_cam: np.ndarray, T_cam_to_frame: np.ndarray) -> np.ndarray:
    return np.asarray(T_cam_to_frame, dtype=np.float64)[:3, :3] @ np.asarray(R_cam, dtype=np.float64)


def _shape_fit_to_descriptor(
    fit_result: Dict[str, Any] | None,
    T_cam_to_frame: np.ndarray,
) -> Dict[str, Any] | None:
    if not fit_result:
        return None
    fit_type = fit_result.get("best_type")
    primitive = fit_result.get("best_result")
    if fit_type is None or primitive is None:
        return None

    desc: Dict[str, Any] = {
        "fit_type": fit_type,
        "family": _shape_type_family(fit_type),
        "confidence": _shape_fit_confidence(fit_result),
        "rotation": None,
        "axis": None,
        "origin": None,
        "endpoints": None,
        "size_vec": None,
        "family_origin": None,
        "family_size_vec": None,
        "scale": _shape_primitive_scale(fit_type, primitive),
        "sign_ambiguous": False,
        "rotation_reliable": False,
    }

    if fit_type == "cuboid":
        size = np.asarray(primitive.size, dtype=np.float64)
        order = np.argsort(size)
        major_idx = int(order[-1])
        desc["origin"] = _shape_transform_point(primitive.center, T_cam_to_frame)
        desc["family_origin"] = desc["origin"]
        desc["size_vec"] = size
        desc["family_size_vec"] = size
        sorted_size = np.sort(size)
        major_axis_distinct = bool((sorted_size[-1] - sorted_size[-2]) / max(sorted_size[-1], 1e-8) > 0.12)
        if major_axis_distinct:
            desc["axis"] = _safe_normalize(_shape_transform_rotation(primitive.rotation[:, major_idx], T_cam_to_frame))
        desc["rotation_reliable"] = bool((sorted_size[-1] - sorted_size[0]) / max(sorted_size[-1], 1e-8) > 0.18)
        if desc["rotation_reliable"]:
            desc["rotation"] = _shape_transform_rotation(primitive.rotation, T_cam_to_frame)
        desc["sign_ambiguous"] = True
        if not major_axis_distinct:
            desc["confidence"] *= 0.7
    elif fit_type == "cylinder":
        desc["origin"] = _shape_transform_point(primitive.center, T_cam_to_frame)
        desc["axis"] = _safe_normalize(_shape_transform_rotation(primitive.axis, T_cam_to_frame))
        desc["endpoints"] = (
            _shape_transform_point(primitive.bottom_center, T_cam_to_frame),
            _shape_transform_point(primitive.top_center, T_cam_to_frame),
        )
        desc["size_vec"] = np.array([primitive.semi_major, primitive.semi_minor, primitive.height], dtype=np.float64)
        desc["family_origin"] = desc["origin"]
        desc["family_size_vec"] = np.array([
            0.5 * (primitive.semi_major + primitive.semi_minor),
            primitive.height,
        ], dtype=np.float64)
        ratio = float(primitive.semi_major / max(primitive.semi_minor, 1e-8))
        desc["rotation_reliable"] = bool(ratio > 1.15)
        if desc["rotation_reliable"]:
            desc["rotation"] = _shape_transform_rotation(primitive.rotation, T_cam_to_frame)
        desc["sign_ambiguous"] = True
    elif fit_type == "frustum":
        small_center = _shape_transform_point(primitive.small_end_center, T_cam_to_frame)
        large_center = _shape_transform_point(primitive.large_end_center, T_cam_to_frame)
        center_mid = 0.5 * (small_center + large_center)
        desc["origin"] = center_mid
        desc["axis"] = _safe_normalize(_shape_transform_rotation(primitive.axis, T_cam_to_frame))
        desc["endpoints"] = (
            small_center,
            large_center,
        )
        desc["size_vec"] = np.array([
            primitive.small_end_major,
            primitive.small_end_minor,
            primitive.large_end_major,
            primitive.large_end_minor,
            primitive.height,
        ], dtype=np.float64)
        desc["family_origin"] = center_mid
        desc["family_size_vec"] = np.array([
            0.25 * (
                primitive.small_end_major + primitive.small_end_minor +
                primitive.large_end_major + primitive.large_end_minor
            ),
            primitive.height,
        ], dtype=np.float64)
        round_ratio = max(
            primitive.small_end_major / max(primitive.small_end_minor, 1e-8),
            primitive.large_end_major / max(primitive.large_end_minor, 1e-8),
        )
        desc["rotation_reliable"] = bool(round_ratio > 1.08)
        if desc["rotation_reliable"]:
            desc["rotation"] = _shape_transform_rotation(primitive.rotation, T_cam_to_frame)
    elif fit_type == "bowl":
        desc["origin"] = _shape_transform_point(primitive.vertex, T_cam_to_frame)
        desc["family_origin"] = desc["origin"]
        desc["axis"] = _safe_normalize(_shape_transform_rotation(primitive.axis, T_cam_to_frame))
        desc["endpoints"] = (
            _shape_transform_point(primitive.vertex, T_cam_to_frame),
            _shape_transform_point(primitive.rim_center, T_cam_to_frame),
        )
        desc["size_vec"] = np.array([primitive.rim_major, primitive.rim_minor, primitive.depth], dtype=np.float64)
        desc["family_size_vec"] = desc["size_vec"]
        ratio = float(primitive.rim_major / max(primitive.rim_minor, 1e-8))
        desc["rotation_reliable"] = bool(ratio > 1.08)
        if desc["rotation_reliable"]:
            desc["rotation"] = _shape_transform_rotation(primitive.rotation, T_cam_to_frame)
    elif fit_type == "plate":
        desc["origin"] = _shape_transform_point(primitive.vertex, T_cam_to_frame)
        desc["family_origin"] = desc["origin"]
        desc["axis"] = _safe_normalize(_shape_transform_rotation(primitive.axis, T_cam_to_frame))
        desc["endpoints"] = (
            _shape_transform_point(primitive.vertex, T_cam_to_frame),
            _shape_transform_point(primitive.rim_center, T_cam_to_frame),
        )
        desc["size_vec"] = np.array([
            primitive.bottom_major,
            primitive.bottom_minor,
            primitive.rim_major,
            primitive.rim_minor,
            primitive.depth,
        ], dtype=np.float64)
        desc["family_size_vec"] = np.array([
            0.5 * (primitive.rim_major + primitive.rim_minor),
            primitive.depth,
        ], dtype=np.float64)
        ratio = float(primitive.rim_major / max(primitive.rim_minor, 1e-8))
        desc["rotation_reliable"] = bool(ratio > 1.08)
        if desc["rotation_reliable"]:
            desc["rotation"] = _shape_transform_rotation(primitive.rotation, T_cam_to_frame)

    if desc["origin"] is None or desc["scale"] <= 1e-6:
        return None
    if desc.get("family_origin") is None:
        desc["family_origin"] = desc["origin"]
    if desc.get("family_size_vec") is None:
        desc["family_size_vec"] = desc["size_vec"]
    return desc


def _build_shape_init_candidates(
    src_desc: Dict[str, Any] | None,
    tgt_desc: Dict[str, Any] | None,
) -> list[tuple[str, np.ndarray]]:
    if src_desc is None or tgt_desc is None:
        return []
    compat = _shape_type_compatibility(src_desc.get("fit_type"), tgt_desc.get("fit_type"))
    conf = min(float(src_desc.get("confidence", 0.0)), float(tgt_desc.get("confidence", 0.0)))
    if compat <= 0.0 or conf < 0.42:
        return []

    items: list[tuple[str, np.ndarray]] = []
    if src_desc.get("family") == tgt_desc.get("family") == "elongated_solid":
        src_origin = np.asarray(src_desc.get("family_origin", src_desc["origin"]), dtype=np.float64)
        tgt_origin = np.asarray(tgt_desc.get("family_origin", tgt_desc["origin"]), dtype=np.float64)
    else:
        src_origin = np.asarray(src_desc["origin"], dtype=np.float64)
        tgt_origin = np.asarray(tgt_desc["origin"], dtype=np.float64)

    if (
        src_desc.get("rotation") is not None
        and tgt_desc.get("rotation") is not None
        and src_desc.get("rotation_reliable")
        and tgt_desc.get("rotation_reliable")
        and src_desc.get("fit_type") in {"frustum", "bowl", "plate"}
        and tgt_desc.get("fit_type") in {"frustum", "bowl", "plate"}
    ):
        R = np.asarray(tgt_desc["rotation"], dtype=np.float64) @ np.asarray(src_desc["rotation"], dtype=np.float64).T
        T = make_transform(R, tgt_origin - R @ src_origin)
        _append_unique_init(items, f"shape_{src_desc['fit_type']}_full", T)

    if src_desc.get("axis") is not None and tgt_desc.get("axis") is not None:
        R_axis = _align_vectors_rotation(np.asarray(src_desc["axis"], dtype=np.float64), np.asarray(tgt_desc["axis"], dtype=np.float64))
        T_axis = make_transform(R_axis, tgt_origin - R_axis @ src_origin)
        _append_unique_init(items, f"shape_{src_desc['fit_type']}_axis", T_axis)

        if bool(src_desc.get("sign_ambiguous")) or bool(tgt_desc.get("sign_ambiguous")):
            R_flip = _align_vectors_rotation(np.asarray(src_desc["axis"], dtype=np.float64), -np.asarray(tgt_desc["axis"], dtype=np.float64))
            T_flip = make_transform(R_flip, tgt_origin - R_flip @ src_origin)
            _append_unique_init(items, f"shape_{src_desc['fit_type']}_axis_flip", T_flip)

    return items


def _shape_similarity_score(
    src_desc: Dict[str, Any] | None,
    tgt_desc: Dict[str, Any] | None,
    T_src_to_tgt: np.ndarray,
    relation_info: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    out = {
        "enabled": False,
        "score": 0.0,
        "weight": 0.0,
        "confidence": 0.0,
        "compatibility": 0.0,
        "center_score": 0.0,
        "axis_score": 0.0,
        "size_score": 0.0,
        "endpoint_score": 0.0,
        "rotation_score": 0.0,
        "relation_score": 0.0,
    }
    if src_desc is None or tgt_desc is None:
        return out

    compat = _shape_type_compatibility(src_desc.get("fit_type"), tgt_desc.get("fit_type"))
    conf = min(float(src_desc.get("confidence", 0.0)), float(tgt_desc.get("confidence", 0.0)))
    if compat <= 0.0 or conf < 0.42:
        return out

    T = np.asarray(T_src_to_tgt, dtype=np.float64).reshape(4, 4)
    scale = max(float(max(src_desc.get("scale", 0.0), tgt_desc.get("scale", 0.0))), 1e-4)
    if src_desc.get("family") == tgt_desc.get("family") == "elongated_solid":
        origin_src = np.asarray(src_desc.get("family_origin", src_desc["origin"]), dtype=np.float64)
        origin_tgt = np.asarray(tgt_desc.get("family_origin", tgt_desc["origin"]), dtype=np.float64)
    else:
        origin_src = np.asarray(src_desc["origin"], dtype=np.float64)
        origin_tgt = np.asarray(tgt_desc["origin"], dtype=np.float64)
    origin_src_t = T[:3, :3] @ origin_src + T[:3, 3]
    center_err = float(np.linalg.norm(origin_src_t - origin_tgt))
    center_score = float(np.exp(-((center_err / max(0.22 * scale, 0.010)) ** 2)))

    axis_score = 0.0
    if src_desc.get("axis") is not None and tgt_desc.get("axis") is not None:
        axis_src_t = _safe_normalize(T[:3, :3] @ np.asarray(src_desc["axis"], dtype=np.float64))
        axis_tgt = _safe_normalize(np.asarray(tgt_desc["axis"], dtype=np.float64))
        dot = float(np.clip(np.dot(axis_src_t, axis_tgt), -1.0, 1.0))
        if bool(src_desc.get("sign_ambiguous")) or bool(tgt_desc.get("sign_ambiguous")):
            axis_score = abs(dot)
        else:
            axis_score = max(dot, 0.0)

    size_score = 0.0
    if src_desc.get("family") == tgt_desc.get("family") == "elongated_solid":
        src_size = src_desc.get("family_size_vec")
        tgt_size = tgt_desc.get("family_size_vec")
    else:
        src_size = src_desc.get("size_vec")
        tgt_size = tgt_desc.get("size_vec")
    if src_size is not None and tgt_size is not None and len(src_size) == len(tgt_size):
        src_size = np.asarray(src_size, dtype=np.float64)
        tgt_size = np.asarray(tgt_size, dtype=np.float64)
        rel = np.abs(src_size - tgt_size) / np.maximum(np.maximum(src_size, tgt_size), 1e-6)
        size_score = float(np.exp(-((float(np.mean(rel)) / 0.30) ** 2)))

    endpoint_score = 0.0
    if src_desc.get("endpoints") is not None and tgt_desc.get("endpoints") is not None:
        src_ep = np.asarray(src_desc["endpoints"], dtype=np.float64)
        tgt_ep = np.asarray(tgt_desc["endpoints"], dtype=np.float64)
        src_ep_t = (T[:3, :3] @ src_ep.T).T + T[:3, 3]
        if relation_info and relation_info.get("available"):
            if relation_info.get("preferred_relation") == "reversed":
                tgt_ep_cmp = tgt_ep[::-1]
            else:
                tgt_ep_cmp = tgt_ep
            ep_err = float(np.mean(np.linalg.norm(src_ep_t - tgt_ep_cmp, axis=1)))
            endpoint_score = float(np.exp(-((ep_err / max(0.16 * scale, 0.010)) ** 2)))
        elif not bool(src_desc.get("sign_ambiguous")) and not bool(tgt_desc.get("sign_ambiguous")):
            ep_err = float(np.mean(np.linalg.norm(src_ep_t - tgt_ep, axis=1)))
            endpoint_score = float(np.exp(-((ep_err / max(0.18 * scale, 0.010)) ** 2)))

    rotation_score = 0.0
    if (
        src_desc.get("rotation") is not None
        and tgt_desc.get("rotation") is not None
        and src_desc.get("rotation_reliable")
        and tgt_desc.get("rotation_reliable")
        and not bool(src_desc.get("sign_ambiguous"))
        and not bool(tgt_desc.get("sign_ambiguous"))
    ):
        R_src_t = T[:3, :3] @ np.asarray(src_desc["rotation"], dtype=np.float64)
        R_tgt = np.asarray(tgt_desc["rotation"], dtype=np.float64)
        dR = R_tgt.T @ R_src_t
        angle = float(np.arccos(np.clip((np.trace(dR) - 1.0) * 0.5, -1.0, 1.0)))
        rotation_score = float(np.exp(-((angle / np.deg2rad(35.0)) ** 2)))

    relation_score = 0.0
    if relation_info and relation_info.get("available") and src_desc.get("axis") is not None and tgt_desc.get("axis") is not None:
        axis_src_t = _safe_normalize(T[:3, :3] @ np.asarray(src_desc["axis"], dtype=np.float64))
        axis_tgt = _safe_normalize(np.asarray(tgt_desc["axis"], dtype=np.float64))
        dot = float(np.clip(np.dot(axis_src_t, axis_tgt), -1.0, 1.0))
        if relation_info.get("preferred_relation") == "reversed":
            relation_score = float(np.clip((1.0 - dot) * 0.5, 0.0, 1.0))
        else:
            relation_score = float(np.clip((1.0 + dot) * 0.5, 0.0, 1.0))

    parts = []
    weights = []
    for score_name, weight in (
        (center_score, 0.30),
        (axis_score, 0.25 if axis_score > 0.0 else 0.0),
        (size_score, 0.20 if size_score > 0.0 else 0.0),
        (endpoint_score, 0.15 if endpoint_score > 0.0 else 0.0),
        (rotation_score, 0.10 if rotation_score > 0.0 else 0.0),
        (relation_score, 0.20 * float(np.clip(relation_info.get("confidence", 0.0), 0.0, 1.0)) if relation_info and relation_score > 0.0 else 0.0),
    ):
        if weight > 0.0:
            parts.append(score_name)
            weights.append(weight)
    geom_score = 0.0 if not weights else float(np.average(parts, weights=weights))
    weight = float(min(0.22, 0.22 * compat * conf))

    out.update({
        "enabled": True,
        "score": geom_score,
        "weight": weight,
        "confidence": conf,
        "compatibility": compat,
        "center_score": center_score,
        "axis_score": axis_score,
        "size_score": size_score,
        "endpoint_score": endpoint_score,
        "rotation_score": rotation_score,
        "relation_score": relation_score,
    })
    return out


def _camera_fit_endpoints(fit_result: Dict[str, Any] | None) -> np.ndarray | None:
    if not fit_result:
        return None
    fit_type = fit_result.get("best_type")
    primitive = fit_result.get("best_result")
    if primitive is None:
        return None
    if fit_type == "cylinder":
        return np.vstack([
            np.asarray(primitive.bottom_center, dtype=np.float64),
            np.asarray(primitive.top_center, dtype=np.float64),
        ])
    if fit_type == "frustum":
        return np.vstack([
            np.asarray(primitive.small_end_center, dtype=np.float64),
            np.asarray(primitive.large_end_center, dtype=np.float64),
        ])
    return None


def _project_points_to_image_float(
    pts_xyz: np.ndarray,
    intr: CameraIntrinsics,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    z = pts[:, 2]
    valid = np.isfinite(z) & (z > 1e-4)
    uv = np.zeros((len(pts), 2), dtype=np.float64)
    if valid.any():
        uv_valid = np.column_stack([
            pts[valid, 0] * intr.fx / z[valid] + intr.cx,
            pts[valid, 1] * intr.fy / z[valid] + intr.cy,
        ])
        uv[valid] = uv_valid
        h, w = image_hw
        valid &= (
            (uv[:, 0] >= 0.0) & (uv[:, 0] < float(w)) &
            (uv[:, 1] >= 0.0) & (uv[:, 1] < float(h))
        )
    return uv, valid


def _axis_profile_signature(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    depth_m: np.ndarray | None,
    intr: CameraIntrinsics,
    endpoints_cam: np.ndarray | None,
    bins: int = 9,
) -> Dict[str, Any] | None:
    if endpoints_cam is None:
        return None
    uv, valid = _project_points_to_image_float(endpoints_cam, intr, image_rgb.shape[:2])
    if len(uv) != 2 or not bool(np.all(valid)):
        return None

    start_uv = uv[0]
    end_uv = uv[1]
    axis_2d = end_uv - start_uv
    axis_len = float(np.linalg.norm(axis_2d))
    if not np.isfinite(axis_len) or axis_len < 28.0:
        return None
    axis_dir = axis_2d / axis_len
    perp_dir = np.array([-axis_dir[1], axis_dir[0]], dtype=np.float64)

    m = np.asarray(mask) > 0
    if m.ndim == 3:
        m = m[:, :, 0]
    ys, xs = np.nonzero(m)
    if len(xs) < 80:
        return None

    pixels = np.column_stack([xs, ys]).astype(np.float64)
    rel = pixels - start_uv
    t = (rel @ axis_dir) / axis_len
    d = rel @ perp_dir
    valid_px = np.isfinite(t) & np.isfinite(d) & (t >= -0.06) & (t <= 1.06)
    if int(valid_px.sum()) < 80:
        return None

    pixels = pixels[valid_px]
    xs = xs[valid_px]
    ys = ys[valid_px]
    t = t[valid_px]
    d = d[valid_px]

    lab_img = cv2.cvtColor(np.asarray(image_rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)
    colors = lab_img[ys, xs].astype(np.float64)
    if depth_m is not None:
        depths = np.asarray(depth_m, dtype=np.float64)[ys, xs]
    else:
        depths = np.zeros(len(xs), dtype=np.float64)

    features = []
    counts = []
    width_ref = max(float(np.percentile(np.abs(d), 95.0) * 2.0), 1.0)
    depth_ref = depths[np.isfinite(depths) & (depths > 0)]
    depth_mean = float(depth_ref.mean()) if len(depth_ref) > 0 else 0.0
    depth_scale = max(float(depth_ref.std()) if len(depth_ref) > 5 else 0.0, 0.015)
    edges = np.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (t >= lo) & (t < hi if hi < 1.0 else t <= hi)
        count = int(sel.sum())
        counts.append(count)
        if count < 10:
            features.append(np.zeros(6, dtype=np.float64))
            continue
        cols = colors[sel]
        lab_mean = cols.mean(axis=0)
        depth_sel = depths[sel]
        valid_depth = depth_sel[np.isfinite(depth_sel) & (depth_sel > 0)]
        depth_feature = 0.0 if len(valid_depth) == 0 else float((valid_depth.mean() - depth_mean) / depth_scale)
        width_feature = float(np.percentile(np.abs(d[sel]), 90.0) * 2.0 / width_ref)
        feat = np.array([
            lab_mean[0] / 255.0,
            (lab_mean[1] - 128.0) / 64.0,
            (lab_mean[2] - 128.0) / 64.0,
            np.clip(depth_feature, -2.0, 2.0) / 2.0,
            np.clip(width_feature, 0.0, 2.0) / 2.0,
            np.clip(count / max(len(xs) / bins, 1.0), 0.0, 2.0) / 2.0,
        ], dtype=np.float64)
        features.append(feat)

    feature_arr = np.vstack(features)
    coverage = float(np.mean(np.asarray(counts) >= 10))
    return {
        "features": feature_arr,
        "counts": counts,
        "coverage": coverage,
        "axis_len_px": axis_len,
        "start_uv": start_uv.tolist(),
        "end_uv": end_uv.tolist(),
    }


def _compare_axis_profile_signatures(
    src_profile: Dict[str, Any] | None,
    tgt_profile: Dict[str, Any] | None,
) -> Dict[str, Any]:
    out = {
        "available": False,
        "preferred_relation": "same",
        "same_score": 0.0,
        "reversed_score": 0.0,
        "confidence": 0.0,
    }
    if not src_profile or not tgt_profile:
        return out

    src_feat = np.asarray(src_profile.get("features"), dtype=np.float64)
    tgt_feat = np.asarray(tgt_profile.get("features"), dtype=np.float64)
    if src_feat.shape != tgt_feat.shape or src_feat.ndim != 2 or src_feat.shape[0] < 4:
        return out

    coverage = min(float(src_profile.get("coverage", 0.0)), float(tgt_profile.get("coverage", 0.0)))
    if coverage < 0.55:
        return out

    mse_same = float(np.mean((src_feat - tgt_feat) ** 2))
    mse_rev = float(np.mean((src_feat - tgt_feat[::-1]) ** 2))
    same_score = float(np.exp(-(mse_same / 0.08)))
    reversed_score = float(np.exp(-(mse_rev / 0.08)))
    pref = "reversed" if reversed_score > same_score else "same"
    confidence = float(abs(reversed_score - same_score) * coverage)
    out.update({
        "available": True,
        "preferred_relation": pref,
        "same_score": same_score,
        "reversed_score": reversed_score,
        "confidence": confidence,
        "coverage": coverage,
    })
    return out


def _orthogonal_unit(v: np.ndarray) -> np.ndarray:
    v = _safe_normalize(v)
    if np.linalg.norm(v) < 1e-8:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(v[0]) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    ortho = np.cross(v, helper)
    return _safe_normalize(ortho)


def _axis_angle_rotation(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    axis = _safe_normalize(axis)
    if np.linalg.norm(axis) < 1e-8:
        return np.eye(3, dtype=np.float64)
    rad = np.deg2rad(float(angle_deg))
    c = float(np.cos(rad))
    s = float(np.sin(rad))
    x, y, z = axis
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + s * K + (1.0 - c) * (K @ K)


def _rotate_transform_about_world_pivot(
    T_init: np.ndarray,
    pivot_world: np.ndarray,
    axis_world: np.ndarray,
    angle_deg: float = 180.0,
) -> np.ndarray:
    pivot = np.asarray(pivot_world, dtype=np.float64).reshape(3)
    dR = _axis_angle_rotation(axis_world, angle_deg)
    rotated_pivot = dR @ pivot
    dT = make_transform(dR, pivot - rotated_pivot)
    return dT @ np.asarray(T_init, dtype=np.float64).reshape(4, 4)


# ============================================================================
# S2M2 stereo → depth
# ============================================================================

def call_s2m2_depth(
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    baseline: float,
    s2m2_url: str = DEFAULT_S2M2_URL,
) -> np.ndarray:
    """Call S2M2 cloud service to compute depth from a stereo pair."""
    tmpdir = tempfile.mkdtemp(prefix="pose_inhand_s2m2_")
    left_path = os.path.join(tmpdir, "left.jpg")
    right_path = os.path.join(tmpdir, "right.jpg")
    cv2.imwrite(left_path, cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(right_path, cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
    return _call_s2m2_depth_api(
        left_img_path=left_path, right_img_path=right_path,
        fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
        baseline=float(baseline), server_url=s2m2_url,
    )


def depth_to_meters(depth_data: np.ndarray, target_hw: tuple | None = None) -> np.ndarray:
    """
    Convert S2M2 output to (H, W) float32 depth in meters.

    S2M2 may return:
      - (H, W, 3): xyz point cloud map → extract z channel
      - (N, 3): unstructured point cloud → not handled here
      - (H, W): already a depth map

    If target_hw is given and sizes differ, resizes to match.
    """
    if depth_data.ndim == 3 and depth_data.shape[2] == 3:
        d = depth_data[:, :, 2].astype(np.float32)
    elif depth_data.ndim == 2:
        d = depth_data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported depth shape: {depth_data.shape}")

    if np.issubdtype(depth_data.dtype, np.integer):
        d *= 0.001

    invalid = ~np.isfinite(d) | (d <= 0)
    d[invalid] = 0.0

    if target_hw is not None:
        th, tw = target_hw
        if d.shape[0] != th or d.shape[1] != tw:
            d = cv2.resize(d, (tw, th), interpolation=cv2.INTER_NEAREST)

    return d


# ============================================================================
# Qwen detection → SAM3 segmentation
# ============================================================================

def _qwen_detect_bbox(
    rgb: np.ndarray,
    text_prompt: str,
) -> Dict[str, Any]:
    """Call Qwen grounding VLM to detect object bbox from a text prompt."""
    h, w = rgb.shape[:2]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr)
    img_b64 = base64.b64encode(buf).decode("utf-8")
    result = _call_qwen_grounding_api(img_b64, text_prompt, w, h)
    return {
        "bbox": result.get("bbox"),
        "status": result.get("qwen_status", "error"),
        "response": result.get("response", ""),
    }


def _sam3_mask_from_bbox(
    rgb: np.ndarray,
    bbox: list,
    sam3_url: str = DEFAULT_SAM3_URL,
) -> np.ndarray:
    """Call SAM3 with a bbox prompt. Returns uint8 mask (0/255)."""
    from PIL import Image as PILImage
    h, w = rgb.shape[:2]
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
    pil_img = PILImage.fromarray(rgb_u8)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=92)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    result = _call_sam3_segment_api(
        image_base64=img_b64, box_prompt=bbox,
        server_url=sam3_url,
    )
    if not result or not result.get("success"):
        raise RuntimeError(f"SAM3 failed: {result}")
    detections = result.get("detections", [])
    if not detections:
        raise RuntimeError("SAM3 returned no detections")

    mask_data = base64.b64decode(detections[0]["mask"])
    mask_img = PILImage.open(io.BytesIO(mask_data)).convert("L")
    if mask_img.size != (w, h):
        mask_img = mask_img.resize((w, h), PILImage.Resampling.NEAREST)
    return (np.array(mask_img) > 0).astype(np.uint8) * 255


def detect_and_segment(
    rgb: np.ndarray,
    bbox: list | None = None,
    text_prompt: str | None = None,
    sam3_url: str = DEFAULT_SAM3_URL,
) -> Tuple[np.ndarray, list | None, Dict[str, Any]]:
    """
    Full detection + segmentation pipeline.

    Priority:
      1. If bbox is given → SAM3 with bbox directly
      2. If text_prompt is given → Qwen detection → get bbox → SAM3 with bbox
      3. Neither → use center 60% of image as default bbox → SAM3

    Returns: (mask_uint8, bbox_used, debug_info)
    """
    debug: Dict[str, Any] = {}
    h, w = rgb.shape[:2]

    if bbox is not None:
        debug["source"] = "manual_bbox"
    elif text_prompt:
        debug["source"] = "qwen_detection"
        qwen_result = _qwen_detect_bbox(rgb, text_prompt)
        debug["qwen_status"] = qwen_result["status"]
        debug["qwen_response"] = qwen_result["response"][:200]
        if qwen_result["bbox"] is not None:
            bbox = qwen_result["bbox"]
            debug["qwen_bbox"] = bbox
        else:
            raise RuntimeError(
                f"Qwen 检测失败 ('{text_prompt}'): {qwen_result['status']} — {qwen_result['response'][:100]}"
            )
    else:
        debug["source"] = "default_center_bbox"
        margin_x, margin_y = int(w * 0.2), int(h * 0.2)
        bbox = [margin_x, margin_y, w - margin_x, h - margin_y]

    debug["bbox_used"] = bbox
    mask = _sam3_mask_from_bbox(rgb, bbox, sam3_url=sam3_url)
    debug["mask_area_px"] = int(np.count_nonzero(mask))
    return mask, bbox, debug


# ============================================================================
# Coarse-to-fine registration
# ============================================================================

def coarse_fine_registration(
    src_pcd, tgt_pcd,
    T_init: np.ndarray,
    after_depth_m: np.ndarray,
    after_rgb: np.ndarray,
    after_intr: CameraIntrinsics,
    T_after_cam_to_frame: np.ndarray,
    cfg: RegistrationConfig,
    use_coarse_fine: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Two-stage hypothesis search + ICP refinement."""
    debug: Dict[str, Any] = {}
    t0 = time.perf_counter()

    if use_coarse_fine:
        coarse_cfg = replace(cfg,
            search_rx_deg=(-60.0, -30.0, 0.0, 30.0, 60.0),
            search_ry_deg=(-60.0, -30.0, 0.0, 30.0, 60.0),
            search_rz_deg=(-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0),
            search_tx_m=(-0.025, 0.0, 0.025),
            search_ty_m=(-0.025, 0.0, 0.025),
            search_tz_m=(-0.025, 0.0, 0.025),
        )
        T_coarse, coarse_info = local_hypothesis_search(
            src_head_pcd_base=src_pcd, tgt_pcd_base=tgt_pcd, T_init=T_init,
            wrist_depth_m=after_depth_m, wrist_rgb=after_rgb,
            wrist_intr=after_intr, T_wrist_to_base=T_after_cam_to_frame,
            cfg=coarse_cfg,
        )
        debug["coarse_score"] = coarse_info.get("score", -1e9)
        debug["coarse_ms"] = int((time.perf_counter() - t0) * 1000)

        t1 = time.perf_counter()
        fine_cfg = replace(cfg,
            search_rx_deg=(-10.0, -5.0, 0.0, 5.0, 10.0),
            search_ry_deg=(-10.0, -5.0, 0.0, 5.0, 10.0),
            search_rz_deg=(-15.0, -7.0, 0.0, 7.0, 15.0),
            search_tx_m=(-0.005, 0.0, 0.005),
            search_ty_m=(-0.005, 0.0, 0.005),
            search_tz_m=(-0.005, 0.0, 0.005),
        )
        T_search, fine_info = local_hypothesis_search(
            src_head_pcd_base=src_pcd, tgt_pcd_base=tgt_pcd, T_init=T_coarse,
            wrist_depth_m=after_depth_m, wrist_rgb=after_rgb,
            wrist_intr=after_intr, T_wrist_to_base=T_after_cam_to_frame,
            cfg=fine_cfg,
        )
        debug["fine_score"] = fine_info.get("score", -1e9)
        debug["fine_ms"] = int((time.perf_counter() - t1) * 1000)
    else:
        T_search, info = local_hypothesis_search(
            src_head_pcd_base=src_pcd, tgt_pcd_base=tgt_pcd, T_init=T_init,
            wrist_depth_m=after_depth_m, wrist_rgb=after_rgb,
            wrist_intr=after_intr, T_wrist_to_base=T_after_cam_to_frame,
            cfg=cfg,
        )
        debug["search_score"] = info.get("score", -1e9)
        debug["search_ms"] = int((time.perf_counter() - t0) * 1000)

    t_icp = time.perf_counter()
    try:
        T_gicp, gicp_res = refine_with_gicp(src_pcd, tgt_pcd, T_search, cfg)
        debug["gicp_fitness"] = float(gicp_res.fitness)
        debug["gicp_inlier_rmse"] = float(gicp_res.inlier_rmse)
    except Exception as e:
        debug["gicp_error"] = str(e)
        T_gicp = T_search
        debug["gicp_fitness"] = 0.0
        debug["gicp_inlier_rmse"] = 1.0
    debug["gicp_ms"] = int((time.perf_counter() - t_icp) * 1000)

    T_final = T_gicp
    if cfg.use_colored_icp:
        t_c = time.perf_counter()
        try:
            T_col, col_res = refine_with_colored_icp(src_pcd, tgt_pcd, T_gicp, cfg)
            T_final = T_col
            debug["colored_icp_fitness"] = float(col_res.fitness)
            debug["colored_icp_inlier_rmse"] = float(col_res.inlier_rmse)
        except Exception as e:
            debug["colored_icp_error"] = str(e).split("\n")[0]
            debug["colored_icp_fitness"] = debug["gicp_fitness"]
            debug["colored_icp_inlier_rmse"] = debug["gicp_inlier_rmse"]
        debug["colored_icp_ms"] = int((time.perf_counter() - t_c) * 1000)

    debug["total_reg_ms"] = int((time.perf_counter() - t0) * 1000)
    return T_final, debug


# ============================================================================
# Quality assessment
# ============================================================================

def assess_quality(gicp_fitness: float, gicp_rmse: float) -> Dict[str, Any]:
    quality, reasons = "good", []
    if gicp_fitness < 0.3:
        quality = "poor"
        reasons.append(f"low fitness ({gicp_fitness:.3f})")
    elif gicp_fitness < 0.5:
        quality = "fair"
        reasons.append(f"moderate fitness ({gicp_fitness:.3f})")
    if gicp_rmse > 0.01:
        quality = "poor"
        reasons.append(f"high RMSE ({gicp_rmse:.4f}m)")
    elif gicp_rmse > 0.005:
        if quality != "poor":
            quality = "fair"
        reasons.append(f"moderate RMSE ({gicp_rmse:.4f}m)")
    return {"quality": quality, "reasons": reasons}


# ============================================================================
# Main pipeline
# ============================================================================

def run_reestimation(
    before_rgb: np.ndarray,
    before_depth: np.ndarray,
    before_mask: np.ndarray,
    after_rgb: np.ndarray,
    after_depth: np.ndarray,
    after_mask: np.ndarray,
    K_wrist: CameraIntrinsics,
    T_cam_to_base_before: np.ndarray | None = None,
    T_cam_to_base_after: np.ndarray | None = None,
    T_wrist_to_ee: np.ndarray | None = None,
    old_obb: OBBState | None = None,
    use_coarse_fine: bool = True,
    cfg: RegistrationConfig | None = None,
) -> ReEstimationResult:
    """
    Main re-estimation entry point.

    Mode 1 (world): both T_cam_to_base provided
        → point clouds transformed to base frame → registration in base frame
        → output: T_reg (base frame), T_slip, updated OBB (if old_obb given)

    Mode 2 (camera): cam_to_base not provided
        → registration in camera frame
        → output: T_reg (camera frame) = T_slip (if T_wrist_to_ee ≈ I)
    """
    if o3d is None:
        raise RuntimeError("open3d is required")
    if cfg is None:
        cfg = RegistrationConfig()

    t_total = time.perf_counter()
    debug: Dict[str, Any] = {}

    has_extrinsics = (T_cam_to_base_before is not None and T_cam_to_base_after is not None)
    mode = "world" if has_extrinsics else "camera"
    debug["mode"] = mode

    # --- Build point clouds in camera frame ---
    t_pcd = time.perf_counter()
    before_pcd_cam = rgbd_to_pointcloud(before_rgb, before_depth, before_mask, K_wrist, cfg)
    after_pcd_cam = rgbd_to_pointcloud(after_rgb, after_depth, after_mask, K_wrist, cfg)
    before_pcd_cam = preprocess_pcd(before_pcd_cam, cfg)
    after_pcd_cam = preprocess_pcd(after_pcd_cam, cfg)
    debug["before_points"] = len(before_pcd_cam.points)
    debug["after_points"] = len(after_pcd_cam.points)
    debug["pcd_ms"] = int((time.perf_counter() - t_pcd) * 1000)

    # --- Optional primitive fitting in camera frame ---
    shape_before_fit: Dict[str, Any] | None = None
    shape_after_fit: Dict[str, Any] | None = None
    K_mat = np.array([
        [K_wrist.fx, 0.0, K_wrist.cx],
        [0.0, K_wrist.fy, K_wrist.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    before_pts_cam = np.asarray(before_pcd_cam.points, dtype=np.float64)
    after_pts_cam = np.asarray(after_pcd_cam.points, dtype=np.float64)
    if len(before_pts_cam) >= 40:
        try:
            shape_before_fit = fit_camera_primitive_from_points(
                before_pts_cam,
                image_mask=before_mask,
                camera_intrinsics=K_mat,
            )
            debug["shape_before_fit"] = serialize_fit_result(shape_before_fit)
            debug["shape_before_confidence"] = _shape_fit_confidence(shape_before_fit)
            debug["_shape_before_fit_raw"] = shape_before_fit
        except Exception as exc:
            debug["shape_before_fit_error"] = str(exc)
    if len(after_pts_cam) >= 40:
        try:
            shape_after_fit = fit_camera_primitive_from_points(
                after_pts_cam,
                image_mask=after_mask,
                camera_intrinsics=K_mat,
            )
            debug["shape_after_fit"] = serialize_fit_result(shape_after_fit)
            debug["shape_after_confidence"] = _shape_fit_confidence(shape_after_fit)
            debug["_shape_after_fit_raw"] = shape_after_fit
        except Exception as exc:
            debug["shape_after_fit_error"] = str(exc)

    # --- Transform to working frame ---
    if has_extrinsics:
        T_before = np.asarray(T_cam_to_base_before, dtype=np.float64).reshape(4, 4)
        T_after = np.asarray(T_cam_to_base_after, dtype=np.float64).reshape(4, 4)
        src_pcd = copy_pcd(before_pcd_cam)
        src_pcd.transform(T_before)
        tgt_pcd = copy_pcd(after_pcd_cam)
        tgt_pcd.transform(T_after)
        # For visibility scoring, need to project back to after-camera frame
        T_scoring_cam = T_after
        T_before_shape_frame = T_before
        T_after_shape_frame = T_after
    else:
        # Stay in camera frame
        src_pcd = copy_pcd(before_pcd_cam)
        tgt_pcd = copy_pcd(after_pcd_cam)
        T_scoring_cam = np.eye(4, dtype=np.float64)
        T_before_shape_frame = np.eye(4, dtype=np.float64)
        T_after_shape_frame = np.eye(4, dtype=np.float64)

    (
        shape_before_active_fit,
        shape_after_active_fit,
        shape_before_desc,
        shape_after_desc,
        shape_prior_pair,
    ) = _select_effective_shape_pair(
        shape_before_fit, shape_after_fit,
        T_before_shape_frame, T_after_shape_frame,
    )
    debug["shape_prior_pair"] = shape_prior_pair
    shape_init_candidates = _build_shape_init_candidates(shape_before_desc, shape_after_desc)
    debug["shape_init_candidates"] = [name for name, _ in shape_init_candidates]

    shape_relation_info: Dict[str, Any] | None = None
    if (
        shape_prior_pair.get("selected")
        and
        shape_before_active_fit is not None and shape_after_active_fit is not None
        and _shape_type_family(shape_before_active_fit.get("best_type")) == "elongated_solid"
        and _shape_type_family(shape_after_active_fit.get("best_type")) == "elongated_solid"
    ):
        before_profile = _axis_profile_signature(
            before_rgb, before_mask, before_depth, K_wrist,
            _camera_fit_endpoints(shape_before_active_fit),
        )
        after_profile = _axis_profile_signature(
            after_rgb, after_mask, after_depth, K_wrist,
            _camera_fit_endpoints(shape_after_active_fit),
        )
        shape_relation_info = _compare_axis_profile_signatures(before_profile, after_profile)
        if shape_relation_info.get("available"):
            debug["shape_relation_info"] = {
                k: v for k, v in shape_relation_info.items()
                if k in {"available", "preferred_relation", "same_score", "reversed_score", "confidence", "coverage"}
            }

    # --- Initialization candidates ---
    src_pts = np.asarray(src_pcd.points)
    tgt_pts = np.asarray(tgt_pcd.points)

    T_inits: list[tuple[str, np.ndarray]] = []
    pca_candidates = pca_initial_transform_candidates(src_pts, tgt_pts)
    for idx, (T_pca, _) in enumerate(pca_candidates[:2]):
        _append_unique_init(T_inits, f"pca_{idx}", T_pca)
        if idx == 0:
            _append_unique_init(
                T_inits,
                f"pca_{idx}_rz180",
                _rotate_transform_about_src_centroid(T_pca, src_pts, axis="z", angle_deg=180.0),
            )

    centroid_T = make_transform(np.eye(3), tgt_pts.mean(axis=0) - src_pts.mean(axis=0))
    _append_unique_init(T_inits, "centroid", centroid_T)
    _append_unique_init(
        T_inits,
        "centroid_rz180",
        _rotate_transform_about_src_centroid(centroid_T, src_pts, axis="z", angle_deg=180.0),
    )
    for name, T_shape in shape_init_candidates:
        _append_unique_init(T_inits, name, T_shape)

    if not T_inits:
        T_inits.append(("pca", pca_initial_transform(src_pts, tgt_pts)))

    # --- After-image depth in meters for scoring ---
    if after_depth.dtype.kind in {"u", "i"}:
        after_depth_m = after_depth.astype(np.float32) / float(cfg.depth_scale)
    else:
        after_depth_m = after_depth.astype(np.float32)

    # --- Run registration with each init, pick best ---
    best_T = np.eye(4, dtype=np.float64)
    best_score = -1e9
    best_init = "none"
    best_reg_debug: Dict[str, Any] = {}

    for name, T_init in T_inits:
        try:
            T_reg, reg_dbg = coarse_fine_registration(
                src_pcd=src_pcd, tgt_pcd=tgt_pcd, T_init=T_init,
                after_depth_m=after_depth_m, after_rgb=after_rgb,
                after_intr=K_wrist, T_after_cam_to_frame=T_scoring_cam,
                cfg=cfg, use_coarse_fine=use_coarse_fine,
            )
            vis_info = visibility_aware_score(
                src_pcd_base=src_pcd, tgt_pcd_base=tgt_pcd, T_srcbase_to_tgtbase=T_reg,
                wrist_depth=after_depth_m, wrist_rgb=after_rgb,
                wrist_intr=K_wrist, T_wrist_to_base=T_scoring_cam, cfg=cfg,
            )
            sil_info = _projected_mask_alignment_metrics(
                src_pcd=src_pcd,
                T_src_to_tgt=T_reg,
                target_mask=after_mask,
                intr=K_wrist,
                T_after_cam_to_frame=T_scoring_cam,
                depth_ref=after_depth_m,
                depth_tol_m=max(cfg.depth_consistency_thresh_m * 4.0, 0.03),
            )
            cloud_info = _pointcloud_overlap_metrics(
                src_pcd=src_pcd,
                tgt_pcd=tgt_pcd,
                T_src_to_tgt=T_reg,
                cfg=cfg,
            )
            shape_info = _shape_similarity_score(
                shape_before_desc, shape_after_desc, T_reg,
                relation_info=shape_relation_info,
            )
            combined_score = _combine_registration_scores(
                vis_info.get("score", -1e9),
                sil_info,
                shape_metrics=shape_info,
                cloud_metrics=cloud_info,
            )
            score_info = {
                **vis_info,
                "silhouette_precision": sil_info["precision"],
                "silhouette_recall": sil_info["recall"],
                "silhouette_iou": sil_info["iou"],
                "silhouette_f1": sil_info["f1"],
                "cloud_overlap_enabled": cloud_info.get("enabled", False),
                "cloud_overlap_weight": cloud_info.get("weight", 0.0),
                "cloud_overlap_score": cloud_info.get("score", 0.0),
                "cloud_overlap_precision": cloud_info.get("precision", 0.0),
                "cloud_overlap_recall": cloud_info.get("recall", 0.0),
                "cloud_overlap_f1": cloud_info.get("f1", 0.0),
                "cloud_overlap_chamfer_m": cloud_info.get("chamfer_m", 0.0),
                "cloud_overlap_robust_m": cloud_info.get("robust_m", 0.0),
                "shape_enabled": shape_info.get("enabled", False),
                "shape_weight": shape_info.get("weight", 0.0),
                "shape_score": shape_info.get("score", 0.0),
                "shape_confidence": shape_info.get("confidence", 0.0),
                "shape_compatibility": shape_info.get("compatibility", 0.0),
                "shape_center_score": shape_info.get("center_score", 0.0),
                "shape_axis_score": shape_info.get("axis_score", 0.0),
                "shape_size_score": shape_info.get("size_score", 0.0),
                "shape_endpoint_score": shape_info.get("endpoint_score", 0.0),
                "shape_rotation_score": shape_info.get("rotation_score", 0.0),
                "shape_relation_score": shape_info.get("relation_score", 0.0),
                "combined_score": combined_score,
            }
            if combined_score > best_score:
                best_score = combined_score
                best_T = T_reg
                best_init = name
                best_reg_debug = {**reg_dbg, **score_info}
        except Exception as e:
            debug[f"init_{name}_error"] = str(e)

    debug["candidate_inits"] = [name for name, _ in T_inits]
    debug["best_init"] = best_init
    debug["best_score"] = best_score
    debug.update({f"reg_{k}": v for k, v in best_reg_debug.items()})

    # --- M3 endpoint-direction flip resolution for elongated objects ---
    if (
        shape_relation_info
        and shape_relation_info.get("available")
        and shape_relation_info.get("confidence", 0.0) >= 0.12
        and shape_before_desc is not None
        and shape_after_desc is not None
        and shape_before_desc.get("axis") is not None
        and shape_after_desc.get("axis") is not None
    ):
        src_axis_t = _safe_normalize(best_T[:3, :3] @ np.asarray(shape_before_desc["axis"], dtype=np.float64))
        tgt_axis = _safe_normalize(np.asarray(shape_after_desc["axis"], dtype=np.float64))
        axis_dot = float(np.clip(np.dot(src_axis_t, tgt_axis), -1.0, 1.0))
        pref = shape_relation_info.get("preferred_relation", "same")
        relation_conflict = (
            (pref == "reversed" and axis_dot > 0.35) or
            (pref == "same" and axis_dot < -0.35)
        )
        if relation_conflict:
            pivot = (
                best_T[:3, :3] @ np.asarray(shape_before_desc["origin"], dtype=np.float64)
                + best_T[:3, 3]
            )
            camera_forward = _safe_normalize(T_scoring_cam[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
            flip_axis = _safe_normalize(np.cross(src_axis_t, camera_forward))
            if np.linalg.norm(flip_axis) < 1e-6:
                flip_axis = _orthogonal_unit(src_axis_t)
            T_flip = _rotate_transform_about_world_pivot(best_T, pivot, flip_axis, angle_deg=180.0)
            try:
                flip_vis = visibility_aware_score(
                    src_pcd_base=src_pcd, tgt_pcd_base=tgt_pcd, T_srcbase_to_tgtbase=T_flip,
                    wrist_depth=after_depth_m, wrist_rgb=after_rgb,
                    wrist_intr=K_wrist, T_wrist_to_base=T_scoring_cam, cfg=cfg,
                )
                flip_sil = _projected_mask_alignment_metrics(
                    src_pcd=src_pcd,
                    T_src_to_tgt=T_flip,
                    target_mask=after_mask,
                    intr=K_wrist,
                    T_after_cam_to_frame=T_scoring_cam,
                    depth_ref=after_depth_m,
                    depth_tol_m=max(cfg.depth_consistency_thresh_m * 4.0, 0.03),
                )
                flip_cloud = _pointcloud_overlap_metrics(
                    src_pcd=src_pcd,
                    tgt_pcd=tgt_pcd,
                    T_src_to_tgt=T_flip,
                    cfg=cfg,
                )
                flip_shape = _shape_similarity_score(
                    shape_before_desc, shape_after_desc, T_flip,
                    relation_info=shape_relation_info,
                )
                flip_score = _combine_registration_scores(
                    flip_vis.get("score", -1e9), flip_sil, shape_metrics=flip_shape, cloud_metrics=flip_cloud,
                )
                debug["m3_flip_candidate_score"] = flip_score
                debug["m3_flip_axis_dot_before"] = axis_dot
                debug["m3_flip_relation_preference"] = pref
                if flip_score > best_score + 0.015 and flip_shape.get("relation_score", 0.0) >= best_reg_debug.get("shape_relation_score", 0.0):
                    best_T = T_flip
                    best_score = flip_score
                    best_init = f"{best_init}+m3_flip"
                    best_reg_debug = {
                        **best_reg_debug,
                        "shape_enabled": flip_shape.get("enabled", False),
                        "shape_weight": flip_shape.get("weight", 0.0),
                        "shape_score": flip_shape.get("score", 0.0),
                        "shape_confidence": flip_shape.get("confidence", 0.0),
                        "shape_compatibility": flip_shape.get("compatibility", 0.0),
                        "shape_center_score": flip_shape.get("center_score", 0.0),
                        "shape_axis_score": flip_shape.get("axis_score", 0.0),
                        "shape_size_score": flip_shape.get("size_score", 0.0),
                        "shape_endpoint_score": flip_shape.get("endpoint_score", 0.0),
                        "shape_rotation_score": flip_shape.get("rotation_score", 0.0),
                        "shape_relation_score": flip_shape.get("relation_score", 0.0),
                        "silhouette_precision": flip_sil["precision"],
                        "silhouette_recall": flip_sil["recall"],
                        "silhouette_iou": flip_sil["iou"],
                        "silhouette_f1": flip_sil["f1"],
                        "cloud_overlap_enabled": flip_cloud.get("enabled", False),
                        "cloud_overlap_weight": flip_cloud.get("weight", 0.0),
                        "cloud_overlap_score": flip_cloud.get("score", 0.0),
                        "cloud_overlap_precision": flip_cloud.get("precision", 0.0),
                        "cloud_overlap_recall": flip_cloud.get("recall", 0.0),
                        "cloud_overlap_f1": flip_cloud.get("f1", 0.0),
                        "cloud_overlap_chamfer_m": flip_cloud.get("chamfer_m", 0.0),
                        "cloud_overlap_robust_m": flip_cloud.get("robust_m", 0.0),
                        "combined_score": flip_score,
                        "m3_flip_applied": True,
                    }
            except Exception as exc:
                debug["m3_flip_error"] = str(exc)

    # --- Shape-center snap refinement for near-miss alignments ---
    if (
        shape_prior_pair.get("selected")
        and shape_before_desc is not None
        and shape_after_desc is not None
        and shape_before_desc.get("origin") is not None
        and shape_after_desc.get("origin") is not None
        and best_reg_debug.get("shape_center_score", 0.0) < 0.72
    ):
        try:
            src_origin = np.asarray(shape_before_desc["origin"], dtype=np.float64)
            tgt_origin = np.asarray(shape_after_desc["origin"], dtype=np.float64)
            src_origin_t = best_T[:3, :3] @ src_origin + best_T[:3, 3]
            center_delta = tgt_origin - src_origin_t
            snap_candidates: list[tuple[str, np.ndarray]] = []

            def _append_snap(name: str, delta_vec: np.ndarray) -> None:
                delta_vec = np.asarray(delta_vec, dtype=np.float64).reshape(3)
                if not np.all(np.isfinite(delta_vec)):
                    return
                if float(np.linalg.norm(delta_vec)) < 1e-5:
                    return
                T_cand = best_T.copy()
                T_cand[:3, 3] += delta_vec
                snap_candidates.append((name, T_cand))

            _append_snap("shape_snap_half", 0.5 * center_delta)
            _append_snap("shape_snap_full", center_delta)
            if shape_after_desc.get("axis") is not None:
                axis_tgt = _safe_normalize(np.asarray(shape_after_desc["axis"], dtype=np.float64))
                delta_axis = float(np.dot(center_delta, axis_tgt)) * axis_tgt
                delta_perp = center_delta - delta_axis
                _append_snap("shape_snap_perp", delta_perp)
                _append_snap("shape_snap_axis_half", 0.5 * delta_axis)

            best_snap: tuple[str, np.ndarray, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], float] | None = None
            for snap_name, T_snap in snap_candidates:
                snap_vis = visibility_aware_score(
                    src_pcd_base=src_pcd, tgt_pcd_base=tgt_pcd, T_srcbase_to_tgtbase=T_snap,
                    wrist_depth=after_depth_m, wrist_rgb=after_rgb,
                    wrist_intr=K_wrist, T_wrist_to_base=T_scoring_cam, cfg=cfg,
                )
                snap_sil = _projected_mask_alignment_metrics(
                    src_pcd=src_pcd,
                    T_src_to_tgt=T_snap,
                    target_mask=after_mask,
                    intr=K_wrist,
                    T_after_cam_to_frame=T_scoring_cam,
                    depth_ref=after_depth_m,
                    depth_tol_m=max(cfg.depth_consistency_thresh_m * 4.0, 0.03),
                )
                snap_cloud = _pointcloud_overlap_metrics(
                    src_pcd=src_pcd,
                    tgt_pcd=tgt_pcd,
                    T_src_to_tgt=T_snap,
                    cfg=cfg,
                )
                snap_shape = _shape_similarity_score(
                    shape_before_desc, shape_after_desc, T_snap,
                    relation_info=shape_relation_info,
                )
                snap_score = _combine_registration_scores(
                    snap_vis.get("score", -1e9), snap_sil, shape_metrics=snap_shape, cloud_metrics=snap_cloud,
                )
                if best_snap is None or snap_score > best_snap[-1]:
                    best_snap = (snap_name, T_snap, snap_vis, snap_sil, snap_shape, snap_cloud, snap_score)

            if best_snap is not None:
                snap_name, T_snap, snap_vis, snap_sil, snap_shape, snap_cloud, snap_score = best_snap
                debug["shape_snap_best_candidate"] = snap_name
                debug["shape_snap_candidate_score"] = snap_score
                debug["shape_snap_center_delta_m"] = float(np.linalg.norm(center_delta))
                center_gain = float(snap_shape.get("center_score", 0.0) - best_reg_debug.get("shape_center_score", 0.0))
                sil_gain = float(snap_sil.get("f1", 0.0) - best_reg_debug.get("silhouette_f1", 0.0))
                if (
                    snap_score > best_score + 0.006
                    and center_gain > 0.08
                    and sil_gain > -0.03
                ):
                    best_T = T_snap
                    best_score = snap_score
                    best_init = f"{best_init}+{snap_name}"
                    best_reg_debug = {
                        **best_reg_debug,
                        "shape_enabled": snap_shape.get("enabled", False),
                        "shape_weight": snap_shape.get("weight", 0.0),
                        "shape_score": snap_shape.get("score", 0.0),
                        "shape_confidence": snap_shape.get("confidence", 0.0),
                        "shape_compatibility": snap_shape.get("compatibility", 0.0),
                        "shape_center_score": snap_shape.get("center_score", 0.0),
                        "shape_axis_score": snap_shape.get("axis_score", 0.0),
                        "shape_size_score": snap_shape.get("size_score", 0.0),
                        "shape_endpoint_score": snap_shape.get("endpoint_score", 0.0),
                        "shape_rotation_score": snap_shape.get("rotation_score", 0.0),
                        "shape_relation_score": snap_shape.get("relation_score", 0.0),
                        "silhouette_precision": snap_sil["precision"],
                        "silhouette_recall": snap_sil["recall"],
                        "silhouette_iou": snap_sil["iou"],
                        "silhouette_f1": snap_sil["f1"],
                        "cloud_overlap_enabled": snap_cloud.get("enabled", False),
                        "cloud_overlap_weight": snap_cloud.get("weight", 0.0),
                        "cloud_overlap_score": snap_cloud.get("score", 0.0),
                        "cloud_overlap_precision": snap_cloud.get("precision", 0.0),
                        "cloud_overlap_recall": snap_cloud.get("recall", 0.0),
                        "cloud_overlap_f1": snap_cloud.get("f1", 0.0),
                        "cloud_overlap_chamfer_m": snap_cloud.get("chamfer_m", 0.0),
                        "cloud_overlap_robust_m": snap_cloud.get("robust_m", 0.0),
                        "combined_score": snap_score,
                        "shape_snap_applied": snap_name,
                    }
        except Exception as exc:
            debug["shape_snap_error"] = str(exc)

    # --- Final local metric refinement around the current best pose ---
    need_micro_refine = (
        best_reg_debug.get("silhouette_f1", 0.0) < 0.90
        or best_reg_debug.get("cloud_overlap_f1", 0.0) < 0.992
        or best_reg_debug.get("shape_score", 0.0) < 0.90
    )
    if need_micro_refine:
        try:
            if shape_after_desc is not None and shape_after_desc.get("axis") is not None:
                axis_main = _safe_normalize(np.asarray(shape_after_desc["axis"], dtype=np.float64))
                axis_perp0 = _orthogonal_unit(axis_main)
                axis_perp1 = _safe_normalize(np.cross(axis_main, axis_perp0))
                refine_axes = [
                    ("axis", axis_main),
                    ("perp0", axis_perp0),
                    ("perp1", axis_perp1),
                ]
            else:
                refine_axes = [
                    ("x", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
                    ("y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
                    ("z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
                ]

            refine_origin = None
            if shape_after_desc is not None:
                ref = shape_after_desc.get("family_origin", shape_after_desc.get("origin"))
                if ref is not None:
                    refine_origin = np.asarray(ref, dtype=np.float64)
            if refine_origin is None:
                refine_origin = np.asarray(tgt_pcd.points, dtype=np.float64).mean(axis=0)

            trans_steps = (
                max(cfg.voxel_size * 0.8, 0.0015),
                max(cfg.voxel_size * 1.6, 0.0030),
            )
            rot_steps = (1.5, 3.0)
            refine_T = best_T.copy()
            refine_debug = dict(best_reg_debug)
            refine_score = float(best_score)
            refine_applied: list[str] = []

            for round_idx in range(2):
                best_round: tuple[str, np.ndarray, Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], float] | None = None
                for axis_name, axis_vec in refine_axes:
                    axis_vec = _safe_normalize(np.asarray(axis_vec, dtype=np.float64))
                    if np.linalg.norm(axis_vec) < 1e-8:
                        continue
                    for step in trans_steps:
                        for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
                            T_cand = refine_T.copy()
                            T_cand[:3, 3] += sign * step * axis_vec
                            cand_vis = visibility_aware_score(
                                src_pcd_base=src_pcd, tgt_pcd_base=tgt_pcd, T_srcbase_to_tgtbase=T_cand,
                                wrist_depth=after_depth_m, wrist_rgb=after_rgb,
                                wrist_intr=K_wrist, T_wrist_to_base=T_scoring_cam, cfg=cfg,
                            )
                            cand_sil = _projected_mask_alignment_metrics(
                                src_pcd=src_pcd,
                                T_src_to_tgt=T_cand,
                                target_mask=after_mask,
                                intr=K_wrist,
                                T_after_cam_to_frame=T_scoring_cam,
                                depth_ref=after_depth_m,
                                depth_tol_m=max(cfg.depth_consistency_thresh_m * 4.0, 0.03),
                            )
                            cand_cloud = _pointcloud_overlap_metrics(
                                src_pcd=src_pcd,
                                tgt_pcd=tgt_pcd,
                                T_src_to_tgt=T_cand,
                                cfg=cfg,
                            )
                            cand_shape = _shape_similarity_score(
                                shape_before_desc, shape_after_desc, T_cand,
                                relation_info=shape_relation_info,
                            )
                            cand_score = _combine_registration_scores(
                                cand_vis.get("score", -1e9),
                                cand_sil,
                                shape_metrics=cand_shape,
                                cloud_metrics=cand_cloud,
                            )
                            candidate_name = f"micro_t_{axis_name}_{sign_name}_{step*1000.0:.1f}mm"
                            if best_round is None or cand_score > best_round[-1]:
                                best_round = (candidate_name, T_cand, cand_vis, cand_sil, cand_shape, cand_cloud, cand_score)
                    for angle in rot_steps:
                        for sign, sign_name in ((-1.0, "minus"), (1.0, "plus")):
                            T_cand = _rotate_transform_about_world_pivot(
                                refine_T, refine_origin, axis_vec, angle_deg=sign * angle,
                            )
                            cand_vis = visibility_aware_score(
                                src_pcd_base=src_pcd, tgt_pcd_base=tgt_pcd, T_srcbase_to_tgtbase=T_cand,
                                wrist_depth=after_depth_m, wrist_rgb=after_rgb,
                                wrist_intr=K_wrist, T_wrist_to_base=T_scoring_cam, cfg=cfg,
                            )
                            cand_sil = _projected_mask_alignment_metrics(
                                src_pcd=src_pcd,
                                T_src_to_tgt=T_cand,
                                target_mask=after_mask,
                                intr=K_wrist,
                                T_after_cam_to_frame=T_scoring_cam,
                                depth_ref=after_depth_m,
                                depth_tol_m=max(cfg.depth_consistency_thresh_m * 4.0, 0.03),
                            )
                            cand_cloud = _pointcloud_overlap_metrics(
                                src_pcd=src_pcd,
                                tgt_pcd=tgt_pcd,
                                T_src_to_tgt=T_cand,
                                cfg=cfg,
                            )
                            cand_shape = _shape_similarity_score(
                                shape_before_desc, shape_after_desc, T_cand,
                                relation_info=shape_relation_info,
                            )
                            cand_score = _combine_registration_scores(
                                cand_vis.get("score", -1e9),
                                cand_sil,
                                shape_metrics=cand_shape,
                                cloud_metrics=cand_cloud,
                            )
                            candidate_name = f"micro_r_{axis_name}_{sign_name}_{angle:.1f}deg"
                            if best_round is None or cand_score > best_round[-1]:
                                best_round = (candidate_name, T_cand, cand_vis, cand_sil, cand_shape, cand_cloud, cand_score)

                if best_round is None:
                    break

                cand_name, cand_T, cand_vis, cand_sil, cand_shape, cand_cloud, cand_score = best_round
                sil_gain = float(cand_sil.get("f1", 0.0) - refine_debug.get("silhouette_f1", 0.0))
                cloud_gain = float(cand_cloud.get("f1", 0.0) - refine_debug.get("cloud_overlap_f1", 0.0))
                shape_gain = float(cand_shape.get("score", 0.0) - refine_debug.get("shape_score", 0.0))
                if (
                    cand_score > refine_score + 0.0025
                    and cand_sil.get("f1", 0.0) >= refine_debug.get("silhouette_f1", 0.0) - 0.015
                    and cand_cloud.get("f1", 0.0) >= refine_debug.get("cloud_overlap_f1", 0.0) - 0.010
                    and (sil_gain > 0.005 or cloud_gain > 0.003 or shape_gain > 0.02)
                ):
                    refine_T = cand_T
                    refine_score = cand_score
                    refine_applied.append(cand_name)
                    refine_debug = {
                        **refine_debug,
                        "shape_enabled": cand_shape.get("enabled", False),
                        "shape_weight": cand_shape.get("weight", 0.0),
                        "shape_score": cand_shape.get("score", 0.0),
                        "shape_confidence": cand_shape.get("confidence", 0.0),
                        "shape_compatibility": cand_shape.get("compatibility", 0.0),
                        "shape_center_score": cand_shape.get("center_score", 0.0),
                        "shape_axis_score": cand_shape.get("axis_score", 0.0),
                        "shape_size_score": cand_shape.get("size_score", 0.0),
                        "shape_endpoint_score": cand_shape.get("endpoint_score", 0.0),
                        "shape_rotation_score": cand_shape.get("rotation_score", 0.0),
                        "shape_relation_score": cand_shape.get("relation_score", 0.0),
                        "silhouette_precision": cand_sil["precision"],
                        "silhouette_recall": cand_sil["recall"],
                        "silhouette_iou": cand_sil["iou"],
                        "silhouette_f1": cand_sil["f1"],
                        "cloud_overlap_enabled": cand_cloud.get("enabled", False),
                        "cloud_overlap_weight": cand_cloud.get("weight", 0.0),
                        "cloud_overlap_score": cand_cloud.get("score", 0.0),
                        "cloud_overlap_precision": cand_cloud.get("precision", 0.0),
                        "cloud_overlap_recall": cand_cloud.get("recall", 0.0),
                        "cloud_overlap_f1": cand_cloud.get("f1", 0.0),
                        "cloud_overlap_chamfer_m": cand_cloud.get("chamfer_m", 0.0),
                        "cloud_overlap_robust_m": cand_cloud.get("robust_m", 0.0),
                        "combined_score": cand_score,
                        "micro_refine_applied": cand_name,
                    }
                else:
                    break

            if refine_applied:
                best_T = refine_T
                best_score = refine_score
                best_init = f"{best_init}+micro_refine"
                best_reg_debug = refine_debug
                debug["micro_refine_sequence"] = refine_applied
        except Exception as exc:
            debug["micro_refine_error"] = str(exc)

    debug["best_init"] = best_init
    debug["best_score"] = best_score
    debug.update({f"reg_{k}": v for k, v in best_reg_debug.items()})

    quality = assess_quality(
        best_reg_debug.get("gicp_fitness", 0.0),
        best_reg_debug.get("gicp_inlier_rmse", 1.0),
    )
    sil_f1 = best_reg_debug.get("silhouette_f1")
    sil_iou = best_reg_debug.get("silhouette_iou")
    if sil_f1 is not None:
        if sil_f1 < 0.20:
            quality["quality"] = "poor"
            quality["reasons"].append(f"very low projected-mask overlap F1 ({sil_f1:.3f})")
        elif sil_f1 < 0.35 and quality["quality"] != "poor":
            quality["quality"] = "fair"
            quality["reasons"].append(f"low projected-mask overlap F1 ({sil_f1:.3f})")
    if sil_iou is not None and sil_iou < 0.12 and quality["quality"] != "poor":
        quality["quality"] = "fair"
        quality["reasons"].append(f"low projected-mask IoU ({sil_iou:.3f})")
    cloud_f1 = best_reg_debug.get("cloud_overlap_f1")
    cloud_chamfer = best_reg_debug.get("cloud_overlap_chamfer_m")
    if cloud_f1 is not None:
        if cloud_f1 < 0.30:
            quality["quality"] = "poor"
            quality["reasons"].append(f"low 3D cloud-overlap F1 ({cloud_f1:.3f})")
        elif cloud_f1 < 0.45 and quality["quality"] != "poor":
            quality["quality"] = "fair"
            quality["reasons"].append(f"moderate 3D cloud-overlap F1 ({cloud_f1:.3f})")
    if cloud_chamfer is not None:
        poor_chamfer = max(cfg.voxel_size * 3.2, 0.011)
        fair_chamfer = max(cfg.voxel_size * 2.2, 0.008)
        if cloud_chamfer > poor_chamfer:
            quality["quality"] = "poor"
            quality["reasons"].append(f"high 3D cloud chamfer ({cloud_chamfer:.4f}m)")
        elif cloud_chamfer > fair_chamfer and quality["quality"] != "poor":
            quality["quality"] = "fair"
            quality["reasons"].append(f"moderate 3D cloud chamfer ({cloud_chamfer:.4f}m)")
    debug["quality"] = quality
    debug["t_slip_frame"] = _t_slip_frame_name(mode, T_wrist_to_ee)

    # --- Compute T_slip ---
    if mode == "world":
        # T_reg is in base frame; convert to gripper-local slip
        T_slip = compute_T_slip(best_T, T_wrist_to_ee)
        new_obb = apply_transform_to_obb(old_obb, best_T) if old_obb is not None else None
    else:
        # T_reg is in camera frame; directly interpretable as slip
        T_slip = compute_T_slip(best_T, T_wrist_to_ee)
        new_obb = None

    elapsed = time.perf_counter() - t_total
    debug["total_elapsed_s"] = elapsed

    return ReEstimationResult(
        success=quality["quality"] != "poor",
        mode=mode,
        T_registration=best_T,
        T_slip=T_slip,
        old_obb=old_obb,
        new_obb=new_obb,
        debug=debug,
        elapsed_s=elapsed,
    )


# ============================================================================
# Visualization helpers
# ============================================================================

def _draw_mask_overlay(img_rgb: np.ndarray, mask: np.ndarray, color=(0, 140, 190)) -> np.ndarray:
    vis = img_rgb.copy()
    m = np.asarray(mask) > 0
    if m.ndim == 3:
        m = m[:, :, 0]
    overlay = vis.copy()
    overlay[m] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(vis, 0.55, overlay, 0.45, 0)


def _draw_detection_overlay(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    bbox: list | None,
    color=(0, 140, 190),
    box_color=(0, 255, 0),
    label: str = "",
) -> np.ndarray:
    vis = _draw_mask_overlay(img_rgb, mask, color=color)
    if bbox:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
        if label:
            cv2.putText(
                vis, label, (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1, cv2.LINE_AA,
            )
    return vis


def _build_context_scene_pcd(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    object_mask: np.ndarray,
    intr: CameraIntrinsics,
    voxel_size: float = 0.006,
):
    if o3d is None:
        return None

    obj = np.asarray(object_mask) > 0
    if obj.ndim == 3:
        obj = obj[:, :, 0]

    cfg_scene = RegistrationConfig(
        depth_scale=1000.0,
        depth_min_m=0.01,
        depth_max_m=3.0,
        erode_kernel=1,
        erode_iters=0,
        remove_depth_edge_points=False,
        voxel_size=max(float(voxel_size), 0.006),
        outlier_nb_neighbors=10,
        outlier_std_ratio=3.0,
        radius_outlier_radius=0.0,
        radius_outlier_min_neighbors=0,
        iqr_multiplier=0.0,
    )

    full_mask = np.ones(depth_m.shape[:2], dtype=np.uint8) * 255
    try:
        pcd = rgbd_to_pointcloud(rgb, depth_m, full_mask, intr, cfg_scene)
    except Exception:
        return None

    if len(pcd.points) == 0:
        return None

    try:
        pcd = pcd.voxel_down_sample(cfg_scene.voxel_size)
    except Exception:
        pass
    return pcd


def _project_points_to_image(
    pts_xyz: np.ndarray,
    intr: CameraIntrinsics,
    image_hw: tuple[int, int],
    depth_ref: np.ndarray | None = None,
    depth_tol_m: float = 0.03,
) -> np.ndarray:
    pts = np.asarray(pts_xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    z = pts[:, 2]
    valid = np.isfinite(z) & (z > 1e-4)
    pts = pts[valid]
    z = z[valid]
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    u = pts[:, 0] * intr.fx / z + intr.cx
    v = pts[:, 1] * intr.fy / z + intr.cy
    ui = np.rint(u).astype(np.int32)
    vi = np.rint(v).astype(np.int32)
    h, w = image_hw
    valid = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
    ui, vi, z = ui[valid], vi[valid], z[valid]
    if len(ui) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    if depth_ref is not None:
        d = depth_ref[vi, ui]
        visible = (d <= 0) | (np.abs(d - z) <= depth_tol_m) | (z <= d + depth_tol_m)
        ui, vi = ui[visible], vi[visible]
    if len(ui) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    return np.stack([ui, vi], axis=1)


def _draw_registration_overlay_2d(
    after_rgb: np.ndarray,
    after_mask: np.ndarray,
    after_depth: np.ndarray,
    intr: CameraIntrinsics,
    after_pcd,
    aligned_pcd,
) -> np.ndarray:
    left = _draw_mask_overlay(after_rgb, after_mask, color=(255, 166, 77))
    right = left.copy()

    m = np.asarray(after_mask) > 0
    if m.ndim == 3:
        m = m[:, :, 0]
    m_u8 = m.astype(np.uint8)
    edge = m_u8 - cv2.erode(m_u8, np.ones((5, 5), np.uint8), iterations=1)
    right[edge > 0] = np.array([255, 166, 77], dtype=np.uint8)
    left[edge > 0] = np.array([255, 166, 77], dtype=np.uint8)

    aligned_pts = np.asarray(aligned_pcd.points)
    aligned_uv = _project_points_to_image(
        aligned_pts, intr, after_rgb.shape[:2], depth_ref=after_depth, depth_tol_m=0.035,
    )
    if len(aligned_uv) > 2500:
        idx = np.random.default_rng(42).choice(len(aligned_uv), 2500, replace=False)
        aligned_uv = aligned_uv[idx]

    for x, y in aligned_uv:
        cv2.circle(right, (int(x), int(y)), 1, (76, 175, 255), -1, cv2.LINE_AA)

    if len(aligned_uv) >= 3:
        hull = cv2.convexHull(aligned_uv.reshape(-1, 1, 2).astype(np.int32))
        cv2.polylines(right, [hull], True, (76, 175, 255), 2, cv2.LINE_AA)

    after_pts = np.asarray(after_pcd.points)
    if len(after_pts) > 0:
        after_center = np.median(after_pts.astype(np.float64), axis=0)
        after_uv = _project_points_to_image(after_center.reshape(1, 3), intr, after_rgb.shape[:2])
        if len(after_uv) > 0:
            x, y = after_uv[0]
            cv2.drawMarker(right, (int(x), int(y)), (255, 166, 77), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.putText(right, "After", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 166, 77), 1, cv2.LINE_AA)

    if len(aligned_pts) > 0:
        aligned_center = np.median(aligned_pts.astype(np.float64), axis=0)
        aligned_center_uv = _project_points_to_image(aligned_center.reshape(1, 3), intr, after_rgb.shape[:2])
        if len(aligned_center_uv) > 0:
            x, y = aligned_center_uv[0]
            cv2.drawMarker(right, (int(x), int(y)), (76, 175, 255), cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
            cv2.putText(right, "Before aligned", (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (76, 175, 255), 1, cv2.LINE_AA)

    cv2.putText(left, "After target mask", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 210), 2, cv2.LINE_AA)
    cv2.putText(right, "After + Before aligned projection", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 210), 2, cv2.LINE_AA)
    cv2.putText(right, "Orange = After target, Blue = Before aligned", (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    divider = np.full((after_rgb.shape[0], 10, 3), 18, dtype=np.uint8)
    return np.concatenate([left, divider, right], axis=1)


def _projected_mask_alignment_metrics(
    src_pcd,
    T_src_to_tgt: np.ndarray,
    target_mask: np.ndarray,
    intr: CameraIntrinsics,
    T_after_cam_to_frame: np.ndarray,
    depth_ref: np.ndarray | None = None,
    depth_tol_m: float = 0.035,
    downsample: int = 4,
) -> Dict[str, float]:
    pts_src = np.asarray(src_pcd.points)
    if len(pts_src) == 0:
        return {"precision": 0.0, "recall": 0.0, "iou": 0.0, "f1": 0.0, "proj_area": 0.0}

    pts_tgt = transform_points(pts_src, np.asarray(T_src_to_tgt, dtype=np.float64).reshape(4, 4))
    T_frame_to_cam = invert_transform(np.asarray(T_after_cam_to_frame, dtype=np.float64).reshape(4, 4))
    pts_cam = transform_points(pts_tgt, T_frame_to_cam)
    uv = _project_points_to_image(
        pts_cam, intr, target_mask.shape[:2], depth_ref=depth_ref, depth_tol_m=depth_tol_m,
    )
    if len(uv) == 0:
        return {"precision": 0.0, "recall": 0.0, "iou": 0.0, "f1": 0.0, "proj_area": 0.0}

    ds = max(1, int(downsample))
    h, w = target_mask.shape[:2]
    hs = max(1, (h + ds - 1) // ds)
    ws = max(1, (w + ds - 1) // ds)
    proj_mask = np.zeros((hs, ws), dtype=np.uint8)
    uvs = np.column_stack([
        np.clip(uv[:, 0] // ds, 0, ws - 1),
        np.clip(uv[:, 1] // ds, 0, hs - 1),
    ]).astype(np.int32)
    proj_mask[uvs[:, 1], uvs[:, 0]] = 255
    if len(uvs) >= 3:
        hull = cv2.convexHull(uvs.reshape(-1, 1, 2))
        cv2.fillConvexPoly(proj_mask, hull, 255)
    proj_mask = cv2.dilate(proj_mask, np.ones((3, 3), np.uint8), iterations=1)

    tgt_mask = (np.asarray(target_mask) > 0).astype(np.uint8)
    if tgt_mask.ndim == 3:
        tgt_mask = tgt_mask[:, :, 0]
    if ds > 1:
        tgt_mask = cv2.resize(tgt_mask, (ws, hs), interpolation=cv2.INTER_NEAREST)
    tgt_mask = (tgt_mask > 0).astype(np.uint8)

    proj_bool = proj_mask > 0
    tgt_bool = tgt_mask > 0
    overlap = int(np.count_nonzero(proj_bool & tgt_bool))
    proj_area = int(np.count_nonzero(proj_bool))
    tgt_area = int(np.count_nonzero(tgt_bool))
    union = int(np.count_nonzero(proj_bool | tgt_bool))

    precision = overlap / max(proj_area, 1)
    recall = overlap / max(tgt_area, 1)
    iou = overlap / max(union, 1)
    f1 = 0.0 if (precision + recall) <= 1e-9 else (2.0 * precision * recall / (precision + recall))
    return {
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "f1": float(f1),
        "proj_area": float(proj_area),
    }


def _subsample_points_evenly(points: np.ndarray, max_points: int = 900) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) <= max_points:
        return pts
    idx = np.linspace(0, len(pts) - 1, max_points).round().astype(np.int64)
    idx = np.clip(idx, 0, len(pts) - 1)
    return pts[idx]


def _pointcloud_overlap_metrics(
    src_pcd,
    tgt_pcd,
    T_src_to_tgt: np.ndarray,
    cfg: RegistrationConfig,
    max_points: int = 900,
) -> Dict[str, float]:
    out = {
        "enabled": False,
        "weight": 0.0,
        "score": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "chamfer_m": 0.0,
        "robust_m": 0.0,
        "point_confidence": 0.0,
    }
    src_pts = _subsample_points_evenly(np.asarray(src_pcd.points, dtype=np.float64), max_points=max_points)
    tgt_pts = _subsample_points_evenly(np.asarray(tgt_pcd.points, dtype=np.float64), max_points=max_points)
    if len(src_pts) < 20 or len(tgt_pts) < 20:
        return out

    src_t = transform_points(src_pts, np.asarray(T_src_to_tgt, dtype=np.float64).reshape(4, 4))
    d2 = np.sum((src_t[:, None, :] - tgt_pts[None, :, :]) ** 2, axis=2)
    nn_src = np.sqrt(np.min(d2, axis=1))
    nn_tgt = np.sqrt(np.min(d2, axis=0))

    tol = max(float(cfg.voxel_size) * 2.8, 0.0065)
    precision = float(np.mean(nn_src <= tol))
    recall = float(np.mean(nn_tgt <= tol))
    f1 = 0.0 if (precision + recall) <= 1e-9 else float(2.0 * precision * recall / (precision + recall))
    chamfer = float(0.5 * (float(np.mean(nn_src)) + float(np.mean(nn_tgt))))
    robust = float(0.5 * (
        float(np.percentile(nn_src, 85.0)) +
        float(np.percentile(nn_tgt, 85.0))
    ))
    dist_score = float(np.exp(-((chamfer / max(tol * 1.35, 1e-6)) ** 2)))
    robust_score = float(np.exp(-((robust / max(tol * 1.85, 1e-6)) ** 2)))
    geom_score = float(0.55 * f1 + 0.30 * dist_score + 0.15 * robust_score)
    point_conf = float(np.clip(min(len(src_pts), len(tgt_pts)) / 220.0, 0.0, 1.0))
    weight = float(0.18 * point_conf)
    out.update({
        "enabled": True,
        "weight": weight,
        "score": geom_score,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "chamfer_m": chamfer,
        "robust_m": robust,
        "point_confidence": point_conf,
    })
    return out


def _pointcloud_has_usable_colors(pcd) -> bool:
    if pcd is None or not pcd.has_colors():
        return False
    colors = np.asarray(pcd.colors, dtype=np.float64)
    points = np.asarray(pcd.points)
    if len(colors) != len(points) or len(colors) == 0:
        return False
    return bool(np.isfinite(colors).all() and float(np.std(colors)) > 1e-4)


def _ply_registration_quality_score(
    cloud_info: Dict[str, float],
    gicp_fitness: float,
    gicp_rmse: float,
    colored_fitness: float | None,
    colored_rmse: float | None,
    cfg: RegistrationConfig,
) -> float:
    cloud_score = float(np.clip(cloud_info.get("score", 0.0), 0.0, 1.0))
    g_rmse = float(gicp_rmse) if np.isfinite(gicp_rmse) else 1e6
    g_rmse_score = float(np.exp(-((g_rmse / max(float(cfg.gicp_max_corr_dist) * 0.45, 1e-6)) ** 2)))
    gicp_score = float(np.clip(gicp_fitness, 0.0, 1.0) * g_rmse_score)
    if colored_fitness is None or colored_rmse is None:
        return float(0.82 * cloud_score + 0.18 * gicp_score)
    c_rmse = float(colored_rmse) if np.isfinite(colored_rmse) else 1e6
    c_rmse_score = float(np.exp(-((c_rmse / max(float(cfg.colored_icp_max_corr_dist) * 0.45, 1e-6)) ** 2)))
    colored_score = float(np.clip(colored_fitness, 0.0, 1.0) * c_rmse_score)
    return float(0.70 * cloud_score + 0.15 * gicp_score + 0.15 * colored_score)


def _build_ply_candidate_summary(rows: list[Dict[str, Any]]) -> str:
    if not rows:
        return "PLY candidate metrics:\n  n/a"
    ordered = sorted(rows, key=lambda row: float(row.get("score", -1e9)), reverse=True)

    def _cell(value: Any, precision: int = 3, scale: float = 1.0, width: int = 8) -> str:
        try:
            val = float(value) * float(scale)
        except (TypeError, ValueError):
            return "n/a".rjust(width)
        if not np.isfinite(val):
            return "n/a".rjust(width)
        return f"{val:.{precision}f}".rjust(width)

    name_width = max(18, min(42, max(len(str(row.get("name", "?"))) for row in ordered)))
    lines = [f"PLY candidate metrics ({len(rows)} total, sorted by score):"]
    lines.append(
        "  "
        f"{'#':>2} "
        f"{'candidate':<{name_width}} "
        f"{'win':>3} "
        f"{'score':>8} "
        f"{'cloudF1':>8} "
        f"{'cham(mm)':>9} "
        f"{'gFit':>8} "
        f"{'gRMSE(mm)':>10} "
        f"{'cFit':>8} "
        f"{'cRMSE(mm)':>10}"
    )
    for rank, row in enumerate(ordered, start=1):
        lines.append(
            "  "
            f"{rank:>2} "
            f"{str(row.get('name', '?')):<{name_width}} "
            f"{'*' if row.get('selected') else '':>3} "
            f"{_cell(row.get('score'))} "
            f"{_cell(row.get('cloud_overlap_f1'))} "
            f"{_cell(row.get('cloud_overlap_chamfer_m'), precision=1, scale=1000.0, width=9)} "
            f"{_cell(row.get('gicp_fitness'))} "
            f"{_cell(row.get('gicp_inlier_rmse_m'), precision=2, scale=1000.0, width=10)} "
            f"{_cell(row.get('colored_icp_fitness'))} "
            f"{_cell(row.get('colored_icp_inlier_rmse_m'), precision=2, scale=1000.0, width=10)}"
        )
    return "\n".join(lines)


def _combine_registration_scores(
    visibility_score: float,
    silhouette_metrics: Dict[str, float],
    shape_metrics: Dict[str, float] | None = None,
    cloud_metrics: Dict[str, float] | None = None,
) -> float:
    vis = float(visibility_score)
    sil_f1 = float(silhouette_metrics.get("f1", 0.0))
    sil_iou = float(silhouette_metrics.get("iou", 0.0))
    sil_recall = float(silhouette_metrics.get("recall", 0.0))
    base_score = 0.50 * vis + 0.30 * sil_f1 + 0.10 * sil_iou + 0.10 * sil_recall
    if cloud_metrics and cloud_metrics.get("enabled"):
        cloud_weight = float(np.clip(cloud_metrics.get("weight", 0.0), 0.0, 0.18))
        cloud_score = float(np.clip(cloud_metrics.get("score", 0.0), 0.0, 1.0))
        base_score = float((1.0 - cloud_weight) * base_score + cloud_weight * cloud_score)
    if not shape_metrics or not shape_metrics.get("enabled"):
        return float(base_score)
    shape_weight = float(np.clip(shape_metrics.get("weight", 0.0), 0.0, 0.22))
    shape_score = float(np.clip(shape_metrics.get("score", 0.0), 0.0, 1.0))
    return float((1.0 - shape_weight) * base_score + shape_weight * shape_score)


def _build_detection_debug_text(
    title: str,
    det_debug: Dict[str, Any],
    bbox: list | None,
    mask: np.ndarray,
    depth: np.ndarray | None = None,
) -> str:
    m = np.asarray(mask) > 0
    if m.ndim == 3:
        m = m[:, :, 0]
    mask_px = int(m.sum())
    lines = [
        f"[{title}]",
        f"source: {det_debug.get('source', '?')}",
        f"bbox: {bbox}",
        f"mask_px: {mask_px}",
    ]
    if depth is not None:
        valid_depth = np.asarray(depth) > 0
        lines.append(f"mask∩depth: {int((m & valid_depth).sum())}")
    if "qwen_status" in det_debug:
        lines.append(f"qwen_status: {det_debug.get('qwen_status')}")
    if det_debug.get("qwen_response"):
        lines.append(f"qwen_response: {str(det_debug.get('qwen_response'))[:180]}")
    return "\n".join(lines)


def _obb_to_plotly_traces(obb: OBBState, color="lime", name="OBB"):
    import plotly.graph_objects as go
    corners = obb.corners_world().astype(np.float64)
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    lx, ly, lz = [], [], []
    for a, b in edges:
        lx += [float(corners[a, 0]), float(corners[b, 0]), None]
        ly += [float(corners[a, 1]), float(corners[b, 1]), None]
        lz += [float(corners[a, 2]), float(corners[b, 2]), None]
    return [go.Scatter3d(x=lx, y=ly, z=lz, mode="lines", line=dict(color=color, width=3), name=name)]


def _pcd_to_plotly(pcd, name="Cloud", max_pts=15000, point_size=3.0, opacity=0.9):
    import plotly.graph_objects as go
    pts = np.asarray(pcd.points)
    if len(pts) == 0:
        return []
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(pts) * 0.5
    if len(pts) > max_pts:
        idx = np.random.default_rng(42).choice(len(pts), max_pts, replace=False)
        pts, cols = pts[idx], cols[idx]
    cs = [f"rgb({int(c[0] * 255)},{int(c[1] * 255)},{int(c[2] * 255)})" for c in cols]
    return [go.Scatter3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
        marker=dict(size=point_size, color=cs, opacity=opacity), name=name,
    )]


_3D_LAYOUT = dict(
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data", bgcolor="#f5f7fa"),
    paper_bgcolor="#fff", margin=dict(l=0, r=0, t=30, b=0), height=520,
)


# ============================================================================
# Summary text
# ============================================================================

def _build_summary(
    result: ReEstimationResult,
    s2m2_before_ms: int = 0, s2m2_after_ms: int = 0,
    sam3_before_ms: int = 0, sam3_after_ms: int = 0,
) -> str:
    lines = []
    q = result.debug.get("quality", {})
    lines.append(f"{'═' * 50}")
    lines.append(f"  模式: {'世界坐标系 (world)' if result.mode == 'world' else '相机坐标系 (camera)'}")
    lines.append(f"  配准质量: {q.get('quality', '?').upper()}")
    for r in q.get("reasons", []):
        lines.append(f"    - {r}")
    lines.append(f"  总耗时: {result.elapsed_s:.3f}s")
    lines.append(f"  最佳初始化: {result.debug.get('best_init', '?')}")
    lines.append(f"{'═' * 50}")

    lines.append("")
    lines.append("服务调用耗时:")
    lines.append(f"  S2M2 (before): {s2m2_before_ms}ms")
    lines.append(f"  S2M2 (after):  {s2m2_after_ms}ms")
    lines.append(f"  SAM3 (before): {sam3_before_ms}ms")
    lines.append(f"  SAM3 (after):  {sam3_after_ms}ms")

    lines.append("")
    lines.append("点云信息:")
    lines.append(f"  Before 点数: {result.debug.get('before_points', '?')}")
    lines.append(f"  After 点数:  {result.debug.get('after_points', '?')}")

    # Init errors
    init_errors = {k: v for k, v in result.debug.items() if k.startswith("init_") and k.endswith("_error")}
    if init_errors:
        lines.append("")
        lines.append("初始化错误:")
        for k, v in init_errors.items():
            name = k.replace("init_", "").replace("_error", "")
            lines.append(f"  {name}: {v}")

    lines.append("")
    lines.append("配准耗时:")
    found_timing = False
    for key in ["coarse_ms", "fine_ms", "search_ms", "gicp_ms", "colored_icp_ms"]:
        val = result.debug.get(f"reg_{key}")
        if val is not None:
            lines.append(f"  {key}: {val}ms")
            found_timing = True
    if not found_timing:
        lines.append("  (无配准数据 — 所有初始化均失败)")

    lines.append("")
    lines.append("ICP 指标:")
    gicp_f = result.debug.get("reg_gicp_fitness")
    gicp_r = result.debug.get("reg_gicp_inlier_rmse")
    if gicp_f is not None:
        lines.append(f"  GICP fitness: {gicp_f:.4f}")
        lines.append(f"  GICP RMSE:    {gicp_r:.6f}m" if gicp_r is not None else "  GICP RMSE:    ?")
    else:
        lines.append("  (无 ICP 数据 — 配准未完成)")
    c = result.debug.get("reg_colored_icp_fitness")
    if c is not None:
        lines.append(f"  Colored ICP fitness: {c}")
        lines.append(f"  Colored ICP RMSE:    {result.debug.get('reg_colored_icp_inlier_rmse')}")
    axis_w = result.debug.get("reg_axis_prior_weight")
    if axis_w is not None and axis_w > 1e-6:
        lines.append(
            "  Elongated-object prior:"
            f" w={axis_w:.3f}, axis={result.debug.get('reg_axis_alignment_score', 0.0):.3f},"
            f" center={result.debug.get('reg_axis_center_score', 0.0):.3f},"
            f" length={result.debug.get('reg_axis_length_score', 0.0):.3f}"
        )
    sil_f1 = result.debug.get("reg_silhouette_f1")
    if sil_f1 is not None:
        lines.append(
            "  Projected mask overlap:"
            f" F1={sil_f1:.3f}, IoU={result.debug.get('reg_silhouette_iou', 0.0):.3f},"
            f" P={result.debug.get('reg_silhouette_precision', 0.0):.3f},"
            f" R={result.debug.get('reg_silhouette_recall', 0.0):.3f}"
        )
    if result.debug.get("reg_cloud_overlap_enabled"):
        lines.append(
            "  3D cloud overlap:"
            f" F1={result.debug.get('reg_cloud_overlap_f1', 0.0):.3f},"
            f" P={result.debug.get('reg_cloud_overlap_precision', 0.0):.3f},"
            f" R={result.debug.get('reg_cloud_overlap_recall', 0.0):.3f},"
            f" chamfer={1000.0 * result.debug.get('reg_cloud_overlap_chamfer_m', 0.0):.1f}mm"
        )
    if result.debug.get("shape_before_fit") or result.debug.get("shape_after_fit"):
        lines.append("")
        lines.append("形状拟合:")
        shape_pair_info = result.debug.get("shape_prior_pair") or {}
        if shape_pair_info.get("selected"):
            lines.append(
                "  Shape prior pair:"
                f" src={shape_pair_info.get('source_type')},"
                f" tgt={shape_pair_info.get('target_type')},"
                f" mode={shape_pair_info.get('selection_mode')},"
                f" compat={shape_pair_info.get('compatibility', 0.0):.3f},"
                f" conf={shape_pair_info.get('confidence', 0.0):.3f}"
            )
            if (
                shape_pair_info.get("source_type") != shape_pair_info.get("source_best_type")
                or shape_pair_info.get("target_type") != shape_pair_info.get("target_best_type")
            ):
                lines.append(
                    "  Shape prior fallback:"
                    f" best=({shape_pair_info.get('source_best_type')}->{shape_pair_info.get('target_best_type')}),"
                    f" selected=({shape_pair_info.get('source_type')}->{shape_pair_info.get('target_type')})"
                )
        if result.debug.get("shape_before_fit"):
            for line in summarize_fit_result_lines(result.debug.get("shape_before_fit"), prefix="  Before"):
                lines.append(f"  {line}" if not line.startswith("  ") else line)
        if result.debug.get("shape_after_fit"):
            for line in summarize_fit_result_lines(result.debug.get("shape_after_fit"), prefix="  After"):
                lines.append(f"  {line}" if not line.startswith("  ") else line)
        if result.debug.get("shape_before_fit_error"):
            lines.append(f"  Before fit error: {result.debug.get('shape_before_fit_error')}")
        if result.debug.get("shape_after_fit_error"):
            lines.append(f"  After fit error: {result.debug.get('shape_after_fit_error')}")
        shape_w = result.debug.get("reg_shape_weight")
        if shape_w is not None and shape_w > 1e-6:
            lines.append(
                "  Shape prior:"
                f" w={shape_w:.3f}, score={result.debug.get('reg_shape_score', 0.0):.3f},"
                f" center={result.debug.get('reg_shape_center_score', 0.0):.3f},"
                f" axis={result.debug.get('reg_shape_axis_score', 0.0):.3f}"
            )
        relation_info = result.debug.get("shape_relation_info") or {}
        if relation_info.get("available"):
            lines.append(
                "  M3 endpoint cue:"
                f" pref={relation_info.get('preferred_relation')},"
                f" conf={relation_info.get('confidence', 0.0):.3f},"
                f" same={relation_info.get('same_score', 0.0):.3f},"
                f" reversed={relation_info.get('reversed_score', 0.0):.3f}"
            )
        if result.debug.get("reg_m3_flip_applied"):
            lines.append("  M3 flip correction: applied")
        if result.debug.get("reg_shape_snap_applied"):
            lines.append(f"  Shape center snap: {result.debug.get('reg_shape_snap_applied')}")
        if result.debug.get("reg_micro_refine_applied"):
            lines.append(f"  Metric micro-refine: {result.debug.get('reg_micro_refine_applied')}")

    frame_name = result.debug.get("t_slip_frame", "camera" if result.mode == "camera" else "base")
    frame_label = {"camera": "相机系", "ee": "夹爪/末端系", "base": "base系"}.get(frame_name, frame_name)
    lines.append("")
    lines.append(f"T_slip ({frame_label}下的物体相对运动):")
    t = result.T_slip[:3, 3]
    try:
        lines.append(f"  平移: [{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}] m")
        lines.extend(_t_slip_rotation_lines(result.T_slip, frame_name, compact=False))
    except ImportError:
        lines.append(f"  平移: [{t[0]:.5f}, {t[1]:.5f}, {t[2]:.5f}] m")

    if result.mode == "world" and result.old_obb is not None and result.new_obb is not None:
        lines.append("")
        lines.append("OBB 变化 (base 系):")
        oc, nc = result.old_obb.center_base, result.new_obb.center_base
        lines.append(f"  旧 center: [{oc[0]:.4f}, {oc[1]:.4f}, {oc[2]:.4f}]")
        lines.append(f"  新 center: [{nc[0]:.4f}, {nc[1]:.4f}, {nc[2]:.4f}]")
        lines.append(f"  偏移: {np.linalg.norm(nc - oc):.4f}m")
    elif result.mode == "camera":
        lines.append("")
        lines.append("提示: camera 模式不输出 base 系 OBB。")
        lines.append("  后续使用 T_slip 时:")
        lines.append("  new_obb = G_current @ T_slip @ G_at_pick⁻¹ @ old_obb")

    return "\n".join(lines)


# ============================================================================
# Parse helpers
# ============================================================================

def _parse_intrinsic_file(file_path: str) -> Dict[str, float]:
    """
    Parse camera intrinsic file.
    Format:
        # fx fy cx cy baseline(m)
        220.066284 215.175922 319.828328 178.033516 0.035018
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 5:
            return {
                "fx": float(parts[0]),
                "fy": float(parts[1]),
                "cx": float(parts[2]),
                "cy": float(parts[3]),
                "baseline": float(parts[4]),
            }
    raise ValueError(f"Cannot parse intrinsic file: expected '# fx fy cx cy baseline(m)' header + values line")


def _parse_matrix(text: str) -> np.ndarray | None:
    text = text.strip()
    if not text:
        return None
    try:
        return np.array(json.loads(text), dtype=np.float64)
    except json.JSONDecodeError:
        vals = [float(v) for v in text.replace(",", " ").split()]
        n = len(vals)
        if n == 16:
            return np.array(vals, dtype=np.float64).reshape(4, 4)
        if n == 9:
            return np.array(vals, dtype=np.float64).reshape(3, 3)
        if n == 3:
            return np.array(vals, dtype=np.float64).reshape(3)
        raise ValueError(f"Cannot parse matrix from {n} values")


# ============================================================================
# Gradio callback
# ============================================================================

def run_pipeline_gradio(
    before_left, before_right,
    after_left, after_right,
    intrinsic_file,
    cam_to_base_before_str, cam_to_base_after_str,
    T_wrist_to_ee_str,
    obb_center_str, obb_axes_str, obb_size_str,
    object_name,
    use_coarse_fine, voxel_size, gicp_max_corr, use_colored_icp,
    depth_scale,
    sam3_url, s2m2_url,
):
    N_OUT = 18
    empty = None

    # Partial results accumulator — always try to show what we computed so far
    partial = {
        "summary": "", "before_mask_vis": None, "after_mask_vis": None,
        "before_depth_vis": None, "after_depth_vis": None,
        "fig_scene_before": None, "fig_scene_after": None,
        "fig_reg_before": None, "fig_reg_after": None, "fig_3d": None,
        "T_slip_str": "", "T_reg_str": "", "raw_json": "",
        "status": "运行中...",
        "T_slip_json": None, "new_obb_json": None, "T_reg_json": None,
        "download_files": None,
    }

    def _make_depth_colormap(depth_m: np.ndarray) -> np.ndarray:
        valid = np.isfinite(depth_m) & (depth_m > 0)
        if not valid.any():
            return np.zeros((*depth_m.shape, 3), dtype=np.uint8)
        d_min, d_max = depth_m[valid].min(), depth_m[valid].max()
        if d_max - d_min < 1e-6:
            d_max = d_min + 1.0
        norm = np.zeros_like(depth_m, dtype=np.uint8)
        norm[valid] = ((depth_m[valid] - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    def _make_depth_mask_diag(depth_m: np.ndarray, mask: np.ndarray, label: str) -> np.ndarray:
        """Depth colormap with mask contour overlaid + stats text."""
        vis = _make_depth_colormap(depth_m)
        m = (np.asarray(mask) > 0)
        if m.ndim == 3:
            m = m[:, :, 0]
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 2)
        valid_d = np.isfinite(depth_m) & (depth_m > 0)
        overlap = int((m & valid_d).sum())
        mask_px = int(m.sum())
        depth_px = int(valid_d.sum())
        d_in_m = depth_m[m]
        finite_in_m = d_in_m[np.isfinite(d_in_m) & (d_in_m > 0)]
        if len(finite_in_m) > 0:
            d_info = f"depth_in_mask: {finite_in_m.min():.3f}~{finite_in_m.max():.3f}m"
        else:
            d_info = "depth_in_mask: NONE"
        stats = f"{label} | mask:{mask_px} depth:{depth_px} overlap:{overlap} | {d_info}"
        cv2.putText(vis, stats, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, f"depth shape: {depth_m.shape}", (5, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def _returns():
        return (
            partial["summary"], partial["before_mask_vis"], partial["after_mask_vis"],
            partial["before_depth_vis"], partial["after_depth_vis"],
            partial["fig_scene_before"], partial["fig_scene_after"],
            partial["fig_reg_before"], partial["fig_reg_after"], partial["fig_3d"],
            partial["T_slip_str"], partial["T_reg_str"], partial["raw_json"],
            partial["status"],
            partial["T_slip_json"], partial["new_obb_json"], partial["T_reg_json"],
            partial["download_files"],
        )

    try:
        if before_left is None or before_right is None:
            partial["summary"] = "请上传抓取前的左右目图像"
            partial["status"] = "缺少输入"
            return _returns()
        if after_left is None or after_right is None:
            partial["summary"] = "请上传抓取后的左右目图像"
            partial["status"] = "缺少输入"
            return _returns()

        h, w = before_left.shape[:2]

        if intrinsic_file is None:
            partial["summary"] = "请上传相机内参文件 (.txt)"
            partial["status"] = "缺少内参文件"
            return _returns()

        intr = _parse_intrinsic_file(str(intrinsic_file))
        fx, fy, cx, cy, baseline = intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["baseline"]
        K = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)

        # --- S2M2 depth estimation ---
        t0 = time.perf_counter()
        before_depth_raw = call_s2m2_depth(
            before_left, before_right, fx, fy, cx, cy, baseline, s2m2_url.strip(),
        )
        s2m2_before_ms = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        after_depth_raw = call_s2m2_depth(
            after_left, after_right, fx, fy, cx, cy, baseline, s2m2_url.strip(),
        )
        s2m2_after_ms = int((time.perf_counter() - t0) * 1000)

        before_depth = depth_to_meters(before_depth_raw, target_hw=(h, w))
        after_h, after_w = after_left.shape[:2]
        after_depth = depth_to_meters(after_depth_raw, target_hw=(after_h, after_w))

        # --- Qwen detection + SAM3 segmentation ---
        obj_prompt = object_name.strip() if object_name and object_name.strip() else None
        t0 = time.perf_counter()
        before_mask, before_bbox_used, before_det_debug = detect_and_segment(
            before_left, text_prompt=obj_prompt, sam3_url=sam3_url.strip(),
        )
        sam3_before_ms = int((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        after_mask, after_bbox_used, after_det_debug = detect_and_segment(
            after_left, text_prompt=obj_prompt, sam3_url=sam3_url.strip(),
        )
        sam3_after_ms = int((time.perf_counter() - t0) * 1000)

        # --- Build mask visualizations (always, even if registration fails) ---
        before_mask_vis = _draw_mask_overlay(before_left, before_mask, color=(0, 180, 0))
        if before_bbox_used:
            x1, y1, x2, y2 = [int(v) for v in before_bbox_used]
            cv2.rectangle(before_mask_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(before_mask_vis, before_det_debug.get("source", ""), (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        partial["before_mask_vis"] = before_mask_vis

        after_mask_vis = _draw_mask_overlay(after_left, after_mask, color=(0, 140, 190))
        if after_bbox_used:
            x1, y1, x2, y2 = [int(v) for v in after_bbox_used]
            cv2.rectangle(after_mask_vis, (x1, y1), (x2, y2), (0, 128, 255), 2)
            cv2.putText(after_mask_vis, after_det_debug.get("source", ""), (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1, cv2.LINE_AA)
        partial["after_mask_vis"] = after_mask_vis

        # --- Depth diagnostics (always) ---
        partial["before_depth_vis"] = _make_depth_mask_diag(before_depth, before_mask, "Before")
        partial["after_depth_vis"] = _make_depth_mask_diag(after_depth, after_mask, "After")

        # --- Scene point cloud (always, before/after separate) ---
        try:
            import plotly.graph_objects as go

            def _build_scene_fig(rgb, depth_m, msk, K_intr, title, edge_color="lime", max_scene=40000):
                traces = []
                rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8) if rgb.dtype != np.uint8 else rgb
                valid = np.isfinite(depth_m) & (depth_m > 0.01) & (depth_m < 5.0)
                m = (msk > 0) if msk.ndim == 2 else (msk[:, :, 0] > 0)

                # Background (outside mask): small, semi-transparent
                bg = valid & ~m
                bys, bxs = np.nonzero(bg)
                if len(bys) > 0:
                    max_bg = max_scene
                    if len(bys) > max_bg:
                        idx = np.random.default_rng(42).choice(len(bys), max_bg, replace=False)
                        bys, bxs = bys[idx], bxs[idx]
                    bzv = depth_m[bys, bxs].astype(np.float64)
                    bxv = (bxs.astype(np.float64) - K_intr.cx) / K_intr.fx * bzv
                    byv = (bys.astype(np.float64) - K_intr.cy) / K_intr.fy * bzv
                    bcs = [f"rgb({rgb_u8[y,x,0]},{rgb_u8[y,x,1]},{rgb_u8[y,x,2]})" for y, x in zip(bys, bxs)]
                    traces.append(go.Scatter3d(
                        x=bxv, y=byv, z=bzv, mode="markers",
                        marker=dict(size=1.0, color=bcs, opacity=0.4), name="背景",
                    ))

                # Object interior: RGB colored, slightly larger
                obj_valid = m & valid
                oys, oxs = np.nonzero(obj_valid)
                if len(oys) > 0:
                    ozv = depth_m[oys, oxs].astype(np.float64)
                    oxv = (oxs.astype(np.float64) - K_intr.cx) / K_intr.fx * ozv
                    oyv = (oys.astype(np.float64) - K_intr.cy) / K_intr.fy * ozv
                    ocs = [f"rgb({rgb_u8[y,x,0]},{rgb_u8[y,x,1]},{rgb_u8[y,x,2]})" for y, x in zip(oys, oxs)]
                    traces.append(go.Scatter3d(
                        x=oxv, y=oyv, z=ozv, mode="markers",
                        marker=dict(size=2.0, color=ocs, opacity=0.9),
                        name=f"物体 ({len(oys)}pts)",
                    ))

                # Mask edge contour: bright color ring
                m_u8 = m.astype(np.uint8)
                eroded = cv2.erode(m_u8, np.ones((5, 5), np.uint8), iterations=1)
                edge = (m_u8 > 0) & (eroded == 0) & valid
                eys, exs = np.nonzero(edge)
                if len(eys) > 0:
                    max_edge = 3000
                    if len(eys) > max_edge:
                        idx = np.random.default_rng(7).choice(len(eys), max_edge, replace=False)
                        eys, exs = eys[idx], exs[idx]
                    ezv = depth_m[eys, exs].astype(np.float64)
                    exv = (exs.astype(np.float64) - K_intr.cx) / K_intr.fx * ezv
                    eyv = (eys.astype(np.float64) - K_intr.cy) / K_intr.fy * ezv
                    traces.append(go.Scatter3d(
                        x=exv, y=eyv, z=ezv, mode="markers",
                        marker=dict(size=3.5, color=edge_color, opacity=1.0),
                        name=f"Mask 边缘 ({len(eys)}pts)",
                    ))

                fig = go.Figure(data=traces)
                fig.update_layout(title=title, **_3D_LAYOUT)
                return fig

            partial["fig_scene_before"] = _build_scene_fig(
                before_left, before_depth, before_mask, K, "Before 场景点云", obj_color="lime",
            )
            partial["fig_scene_after"] = _build_scene_fig(
                after_left, after_depth, after_mask, K, "After 场景点云", obj_color="orange",
            )
        except Exception:
            pass

        diag_lines = []
        diag_lines.append(f"S2M2 before raw shape: {before_depth_raw.shape}, dtype: {before_depth_raw.dtype}")
        diag_lines.append(f"S2M2 after  raw shape: {after_depth_raw.shape}, dtype: {after_depth_raw.dtype}")
        diag_lines.append(f"Depth before (meters): shape={before_depth.shape}, valid={int((before_depth > 0).sum())}, range=[{before_depth[before_depth > 0].min() if (before_depth > 0).any() else 0:.4f}, {before_depth.max():.4f}]")
        diag_lines.append(f"Depth after  (meters): shape={after_depth.shape}, valid={int((after_depth > 0).sum())}, range=[{after_depth[after_depth > 0].min() if (after_depth > 0).any() else 0:.4f}, {after_depth.max():.4f}]")
        diag_lines.append(f"Before mask pixels: {int((before_mask > 0).sum())}, bbox: {before_bbox_used}")
        diag_lines.append(f"After  mask pixels: {int((after_mask > 0).sum())}, bbox: {after_bbox_used}")
        bm = before_mask > 0 if before_mask.ndim == 2 else before_mask[:, :, 0] > 0
        am = after_mask > 0 if after_mask.ndim == 2 else after_mask[:, :, 0] > 0
        diag_lines.append(f"Before mask∩depth: {int((bm & (before_depth > 0)).sum())}")
        diag_lines.append(f"After  mask∩depth: {int((am & (after_depth > 0)).sum())}")
        diag_lines.append(f"S2M2 耗时: before={s2m2_before_ms}ms, after={s2m2_after_ms}ms")
        diag_lines.append(f"检测+分割耗时: before={sam3_before_ms}ms, after={sam3_after_ms}ms")
        partial["summary"] = "\n".join(diag_lines)

        # --- Parse optional matrices ---
        T_before = _parse_matrix(cam_to_base_before_str)
        T_after = _parse_matrix(cam_to_base_after_str)
        T_we = _parse_matrix(T_wrist_to_ee_str)

        old_obb = None
        if obb_center_str.strip() and obb_size_str.strip():
            axes_raw = _parse_matrix(obb_axes_str) if obb_axes_str.strip() else np.eye(3, dtype=np.float64)
            old_obb = OBBState(
                center_base=_parse_matrix(obb_center_str).reshape(3),
                axes_base=axes_raw.reshape(3, 3),
                size_xyz=_parse_matrix(obb_size_str).reshape(3),
            )

        cfg = RegistrationConfig(
            depth_scale=float(depth_scale),
            voxel_size=float(voxel_size),
            gicp_max_corr_dist=float(gicp_max_corr),
            use_colored_icp=bool(use_colored_icp),
        )

        # --- Run registration ---
        result = run_reestimation(
            before_rgb=before_left,
            before_depth=before_depth,
            before_mask=before_mask,
            after_rgb=after_left,
            after_depth=after_depth,
            after_mask=after_mask,
            K_wrist=K,
            T_cam_to_base_before=T_before,
            T_cam_to_base_after=T_after,
            T_wrist_to_ee=T_we,
            old_obb=old_obb,
            use_coarse_fine=bool(use_coarse_fine),
            cfg=cfg,
        )

        partial["summary"] = _build_summary(result, s2m2_before_ms, s2m2_after_ms, sam3_before_ms, sam3_after_ms)

        # --- 3D registration result (separate before/after + merged) ---
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            cfg2 = RegistrationConfig(voxel_size=float(voxel_size))
            bp = rgbd_to_pointcloud(before_left, before_depth, before_mask, K, cfg2)
            bp = preprocess_pcd(bp, cfg2)
            ap = rgbd_to_pointcloud(after_left, after_depth, after_mask, K, cfg2)
            ap = preprocess_pcd(ap, cfg2)
            if T_before is not None and T_after is not None:
                bp_vis = copy_pcd(bp); bp_vis.transform(np.asarray(T_before, dtype=np.float64).reshape(4, 4))
                ap_vis = copy_pcd(ap); ap_vis.transform(np.asarray(T_after, dtype=np.float64).reshape(4, 4))
            else:
                bp_vis, ap_vis = copy_pcd(bp), copy_pcd(ap)
            aligned = copy_pcd(bp_vis)
            aligned.transform(result.T_registration)

            def _pcd_uniform_trace(pcd, name, color, size=3.5, opacity=0.9):
                pts = np.asarray(pcd.points)
                if len(pts) == 0:
                    return []
                return [go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers",
                    marker=dict(size=size, color=color, opacity=opacity), name=name,
                )]

            # Before aligned (standalone, RGB colors)
            fig_before = go.Figure(data=_pcd_to_plotly(aligned, "Before (配准后)", point_size=3.5))
            fig_before.update_layout(title="Before 物体 (配准后位置) — RGB 颜色", **_3D_LAYOUT)
            partial["fig_reg_before"] = fig_before

            # After (standalone, RGB colors)
            fig_after = go.Figure(data=_pcd_to_plotly(ap_vis, "After", point_size=3.5))
            fig_after.update_layout(title="After 物体 — RGB 颜色", **_3D_LAYOUT)
            partial["fig_reg_after"] = fig_after

            # Merged overlay — fixed colors so you can tell them apart
            # Blue = Before (配准后), Red = After
            traces = _pcd_uniform_trace(aligned, "Before (配准后) — 蓝色", color="dodgerblue", size=3.0, opacity=0.8)
            traces += _pcd_uniform_trace(ap_vis, "After — 红色", color="red", size=3.0, opacity=0.8)
            if result.old_obb:
                traces += _obb_to_plotly_traces(result.old_obb, "lime", "Pre-grasp OBB")
            if result.new_obb:
                traces += _obb_to_plotly_traces(result.new_obb, "orange", "Post-grasp OBB")
            partial["fig_3d"] = go.Figure(data=traces)
            partial["fig_3d"].update_layout(
                title="配准叠加: 蓝=Before(配准后), 红=After — 重合=配准成功, 分离=配准失败",
                **_3D_LAYOUT,
            )
        except Exception:
            pass

        # --- Save PLY files for download ---
        try:
            dl_dir = tempfile.mkdtemp(prefix="pose_inhand_dl_")
            dl_files = []

            def _save_ply(pcd, name):
                path = os.path.join(dl_dir, name)
                o3d.io.write_point_cloud(path, pcd)
                dl_files.append(path)

            # Scene point clouds (full, cleaned)
            cfg_clean = RegistrationConfig(
                depth_scale=float(depth_scale), voxel_size=float(voxel_size),
                depth_min_m=0.01, depth_max_m=5.0,
            )

            def _full_scene_pcd(rgb, depth_m, K_intr):
                h_d, w_d = depth_m.shape[:2]
                full_mask = np.ones((h_d, w_d), dtype=np.uint8) * 255
                K_full = CameraIntrinsics(fx=K_intr.fx, fy=K_intr.fy, cx=K_intr.cx, cy=K_intr.cy, width=w_d, height=h_d)
                pcd = rgbd_to_pointcloud(rgb, depth_m, full_mask, K_full, cfg_clean)
                return preprocess_pcd(pcd, cfg_clean)

            try:
                scene_before = _full_scene_pcd(before_left, before_depth, K)
                _save_ply(scene_before, "scene_before.ply")
            except Exception:
                pass
            try:
                scene_after = _full_scene_pcd(after_left, after_depth, K)
                _save_ply(scene_after, "scene_after.ply")
            except Exception:
                pass

            # Object-only point clouds
            try:
                obj_before = rgbd_to_pointcloud(before_left, before_depth, before_mask, K, cfg)
                obj_before = preprocess_pcd(obj_before, cfg)
                _save_ply(obj_before, "object_before.ply")
            except Exception:
                pass
            try:
                obj_after = rgbd_to_pointcloud(after_left, after_depth, after_mask, K, cfg)
                obj_after = preprocess_pcd(obj_after, cfg)
                _save_ply(obj_after, "object_after.ply")
            except Exception:
                pass

            # Aligned object (after registration)
            try:
                obj_b = rgbd_to_pointcloud(before_left, before_depth, before_mask, K, cfg)
                obj_b = preprocess_pcd(obj_b, cfg)
                obj_b.transform(result.T_registration)
                obj_b.paint_uniform_color([1.0, 0.3, 0.3])
                obj_a = rgbd_to_pointcloud(after_left, after_depth, after_mask, K, cfg)
                obj_a = preprocess_pcd(obj_a, cfg)
                merged = o3d.geometry.PointCloud()
                merged.points = o3d.utility.Vector3dVector(
                    np.vstack([np.asarray(obj_b.points), np.asarray(obj_a.points)]))
                merged.colors = o3d.utility.Vector3dVector(
                    np.vstack([np.asarray(obj_b.colors), np.asarray(obj_a.colors)]))
                _save_ply(merged, "aligned_merged.ply")
            except Exception:
                pass

            # Save JSON result
            json_path = os.path.join(dl_dir, "result.json")
            with open(json_path, "w") as jf:
                jf.write(partial.get("raw_json", "{}") or "{}")
            dl_files.append(json_path)

            # Pack everything into a zip for one-click download
            if dl_files:
                import zipfile
                zip_path = os.path.join(dl_dir, "all_results.zip")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in dl_files:
                        zf.write(f, os.path.basename(f))
                dl_files.append(zip_path)

            partial["download_files"] = dl_files if dl_files else None
        except Exception:
            partial["download_files"] = None

        partial["T_slip_str"] = np.array2string(result.T_slip, precision=6, suppress_small=True)
        partial["T_reg_str"] = np.array2string(result.T_registration, precision=6, suppress_small=True)
        partial["T_slip_json"] = result.T_slip.tolist()
        partial["new_obb_json"] = result.new_obb.to_dict() if result.new_obb else None
        partial["T_reg_json"] = result.T_registration.tolist()

        partial["raw_json"] = json.dumps({
            "success": result.success, "mode": result.mode,
            "elapsed_s": result.elapsed_s,
            "quality": result.debug.get("quality"),
            "T_slip": result.T_slip.tolist(),
            "T_registration": result.T_registration.tolist(),
            "old_obb": result.old_obb.to_dict() if result.old_obb else None,
            "new_obb": result.new_obb.to_dict() if result.new_obb else None,
            "detection_before": before_det_debug, "detection_after": after_det_debug,
            "timings": {"s2m2_before_ms": s2m2_before_ms, "s2m2_after_ms": s2m2_after_ms,
                        "detect_seg_before_ms": sam3_before_ms, "detect_seg_after_ms": sam3_after_ms},
        }, indent=2, ensure_ascii=False, default=str)

        q_info = result.debug.get("quality", {})
        partial["status"] = f"完成 | {q_info.get('quality', '?').upper()} | 模式: {result.mode} | 耗时 {result.elapsed_s:.2f}s"

        return _returns()

    except Exception as e:
        tb = traceback.format_exc()
        partial["summary"] = f"Pipeline 失败:\n{e}\n\n{tb}\n\n--- 已有诊断信息 ---\n{partial['summary']}"
        partial["status"] = f"失败: {e}"
        return _returns()


# ============================================================================
# Gradio UI
# ============================================================================

def build_app():
    if gr is None:
        raise ImportError("gradio is required for web UI. Use --local/--local-files or install gradio.")
    with gr.Blocks(title="Post-Grasp In-Hand Pose Re-Estimation") as app:

        gr.Markdown("# Post-Grasp In-Hand Pose Re-Estimation", elem_id="app-title")

        with gr.Row(equal_height=False):

            # ── Left: Inputs ──
            with gr.Column(scale=3, min_width=380):

                # ───── 模式选择（最顶部） ─────
                mode_radio = gr.Radio(
                    label="工作模式",
                    choices=["camera 模式 (无外参, 输出 T_slip)", "world 模式 (有外参, 输出新 OBB)"],
                    value="camera 模式 (无外参, 输出 T_slip)",
                )

                # ───── 目标物体名称 ─────
                object_name = gr.Textbox(
                    label="目标物体名称 (用于 Qwen 检测 → SAM3 分割)",
                    placeholder="e.g. pen, white card, 笔, 螺丝刀",
                    max_lines=1,
                )

                # ───── 图像输入 ─────
                with gr.Accordion("抓取前 双目图像", open=True):
                    with gr.Row():
                        before_left = gr.Image(label="抓取前 左图", type="numpy", height=150)
                        before_right = gr.Image(label="抓取前 右图", type="numpy", height=150)

                with gr.Accordion("抓取后 双目图像", open=True):
                    with gr.Row():
                        after_left = gr.Image(label="抓取后 左图", type="numpy", height=150)
                        after_right = gr.Image(label="抓取后 右图", type="numpy", height=150)

                # ───── 相机内参 ─────
                intrinsic_file = gr.File(
                    label="上传内参文件 (.txt, 格式: # fx fy cx cy baseline(m))",
                    file_types=[".txt"], type="filepath",
                )
                intrinsic_status = gr.Textbox(label="解析结果", interactive=False, lines=1)

                # ───── 配准参数 ─────
                with gr.Accordion("配准参数", open=False):
                    with gr.Row():
                        depth_scale = gr.Number(label="Depth Scale", value=1.0, precision=1)
                        voxel_size = gr.Number(label="Voxel Size (m)", value=0.0015, precision=4)
                    with gr.Row():
                        gicp_max_corr = gr.Number(label="GICP Max Corr (m)", value=0.01, precision=4)
                        use_colored_icp = gr.Checkbox(label="Colored ICP", value=True)
                    use_coarse_fine = gr.Checkbox(label="Coarse-to-Fine 搜索 (扩大范围到±90°)", value=True)

                # ───── World 模式专属参数 ─────
                with gr.Accordion("外参 cam_to_base (world 模式)", open=False, visible=False) as extrinsic_group:
                    cam_to_base_before_str = gr.Textbox(
                        label="抓取前 cam_to_base (4x4 JSON)", value="", max_lines=2,
                    )
                    cam_to_base_after_str = gr.Textbox(
                        label="抓取后 cam_to_base (4x4 JSON)", value="", max_lines=2,
                    )

                with gr.Accordion("手眼标定 T_wrist_to_ee (可选)", open=False):
                    gr.Markdown("相机到末端执行器的固定变换。留空则 T_slip = T_reg")
                    T_wrist_to_ee_str = gr.Textbox(label="T_wrist_to_ee (4x4)", value="", max_lines=2)

                with gr.Accordion("OBB (world 模式, 可选)", open=False, visible=False) as obb_group:
                    gr.Markdown("world 模式下，配合 OBB 数据输出更新后的 base 系 OBB")
                    obb_center_str = gr.Textbox(label="center_base [x,y,z]", value="", max_lines=1)
                    obb_axes_str = gr.Textbox(label="axes_base 3x3 (留空=单位阵)", value="", max_lines=1)
                    obb_size_str = gr.Textbox(label="size_xyz [L,W,H]", value="", max_lines=1)

                # ───── 服务地址 ─────
                with gr.Accordion("服务地址", open=False):
                    sam3_url = gr.Textbox(label="SAM3 URL", value=DEFAULT_SAM3_URL, max_lines=1)
                    s2m2_url = gr.Textbox(label="S2M2 URL", value=DEFAULT_S2M2_URL, max_lines=1)

            # ── Right: Results ──
            with gr.Column(scale=5, min_width=600):

                run_btn = gr.Button("运行 Re-Estimation", variant="primary", size="lg")
                status_bar = gr.Textbox(label="状态", value="就绪", interactive=False, max_lines=1)

                with gr.Tabs():
                    with gr.Tab("概览"):
                        summary_text = gr.Textbox(
                            label="结果摘要", lines=32, interactive=False,
                            elem_classes=["summary-box"],
                        )

                    with gr.Tab("Mask + Depth 诊断"):
                        gr.Markdown("**Mask overlay** (bbox + 分割) 和 **Depth colormap** (mask 轮廓叠加, 黄色=mask边界)")
                        with gr.Row():
                            before_mask_vis = gr.Image(label="Before: Mask", height=280)
                            after_mask_vis = gr.Image(label="After: Mask", height=280)
                        with gr.Row():
                            before_depth_vis = gr.Image(label="Before: Depth + Mask 轮廓", height=280)
                            after_depth_vis = gr.Image(label="After: Depth + Mask 轮廓", height=280)

                    with gr.Tab("场景点云 (配准前)"):
                        gr.Markdown("全场景 + 物体高亮（绿=物体）。上下分开显示 Before / After")
                        plot_scene_before = gr.Plot(label="Before 场景点云")
                        plot_scene_after = gr.Plot(label="After 场景点云")

                    with gr.Tab("3D 配准结果"):
                        gr.Markdown("上方分别展示 Before/After 物体原始形状，下方叠加显示（重合=配准成功）")
                        with gr.Row():
                            plot_reg_before = gr.Plot(label="Before 物体 (配准后)")
                            plot_reg_after = gr.Plot(label="After 物体")
                        plot_3d = gr.Plot(label="配准叠加")

                    with gr.Tab("T_slip / T_reg"):
                        T_slip_text = gr.Textbox(
                            label="T_slip (物体在夹爪系的滑动, 4x4)", lines=6, interactive=False,
                        )
                        T_reg_text = gr.Textbox(
                            label="T_registration (配准变换, 4x4)", lines=6, interactive=False,
                        )

                    with gr.Tab("输出数据"):
                        with gr.Row():
                            T_slip_json = gr.JSON(label="T_slip (可复制)")
                            new_obb_json = gr.JSON(label="新 OBB (仅 world 模式)")
                        T_reg_json = gr.JSON(label="T_registration")

                    with gr.Tab("下载"):
                        gr.Markdown(
                            "点击文件名下载。包含：场景点云 PLY、物体点云 PLY、配准后合并点云 PLY、结果 JSON。"
                            "  \n用 **MeshLab / CloudCompare** 打开 PLY 文件查看。"
                        )
                        download_files = gr.File(label="可下载文件", file_count="multiple")

                    with gr.Tab("原始 JSON"):
                        raw_json = gr.Code(label="完整输出", language="json", lines=30)

        # ── Mode switch → show/hide world-mode panels ──
        def _on_mode_change(mode):
            is_world = "world" in mode
            return gr.update(visible=is_world, open=is_world), gr.update(visible=is_world)

        mode_radio.change(
            fn=_on_mode_change,
            inputs=[mode_radio],
            outputs=[extrinsic_group, obb_group],
        )

        # ── Intrinsic file upload → show parsed info ──
        def _on_intrinsic_upload(file_path):
            if file_path is None:
                return "未上传"
            try:
                intr = _parse_intrinsic_file(str(file_path))
                return f"fx={intr['fx']:.4f}  fy={intr['fy']:.4f}  cx={intr['cx']:.4f}  cy={intr['cy']:.4f}  baseline={intr['baseline']:.6f}m"
            except Exception as e:
                return f"解析失败: {e}"

        intrinsic_file.change(
            fn=_on_intrinsic_upload,
            inputs=[intrinsic_file],
            outputs=[intrinsic_status],
        )

        # ── Wire up ──
        run_btn.click(
            fn=run_pipeline_gradio,
            inputs=[
                before_left, before_right,
                after_left, after_right,
                intrinsic_file,
                cam_to_base_before_str, cam_to_base_after_str,
                T_wrist_to_ee_str,
                obb_center_str, obb_axes_str, obb_size_str,
                object_name,
                use_coarse_fine, voxel_size, gicp_max_corr, use_colored_icp,
                depth_scale,
                sam3_url, s2m2_url,
            ],
            outputs=[
                summary_text, before_mask_vis, after_mask_vis,
                before_depth_vis, after_depth_vis,
                plot_scene_before, plot_scene_after,
                plot_reg_before, plot_reg_after, plot_3d,
                T_slip_text, T_reg_text, raw_json,
                status_bar,
                T_slip_json, new_obb_json, T_reg_json,
                download_files,
            ],
        )

    return app


# ============================================================================
# Local CLI mode (tkinter software renderer, same style as pointcloud_viewer.py)
# ============================================================================

class _LocalResultViewer:
    """Lightweight tkinter viewer with clearer registration views."""

    BG_RGB = (12, 18, 24)
    BEFORE_RGB = np.array([70, 205, 255], dtype=np.float32) / 255.0
    AFTER_RGB = np.array([255, 186, 56], dtype=np.float32) / 255.0
    BEFORE_ORIG_RGB = np.array([120, 255, 150], dtype=np.float32) / 255.0
    TRAIL_RGB = (150, 235, 255)
    OBJECT_MAX_PTS = 22000
    SCENE_MAX_PTS = 14000

    def __init__(self, before_pcd, after_pcd, aligned_pcd, after_scene_pcd=None, info_text: str = "", parent=None):
        import tkinter as _tk
        from PIL import ImageTk as _ImageTk

        self._tk = _tk
        self._ImageTk = _ImageTk

        self.before_pts = np.asarray(before_pcd.points).astype(np.float32)
        self.after_pts = np.asarray(after_pcd.points).astype(np.float32)
        self.aligned_pts = np.asarray(aligned_pcd.points).astype(np.float32)
        self.before_pts = self._subsample(self.before_pts, self.OBJECT_MAX_PTS)
        self.after_pts = self._subsample(self.after_pts, self.OBJECT_MAX_PTS)
        self.aligned_pts = self._subsample(self.aligned_pts, self.OBJECT_MAX_PTS)

        focus_obj = self._stack_nonempty(self.after_pts, self.aligned_pts)
        self.object_center = self._centroid(focus_obj)
        self.object_diag = max(self._diag(focus_obj), 0.035)

        if after_scene_pcd is not None and len(after_scene_pcd.points) > 0:
            scene_pts = np.asarray(after_scene_pcd.points).astype(np.float32)
            if after_scene_pcd.has_colors():
                scene_cols = np.asarray(after_scene_pcd.colors).astype(np.float32)
            else:
                scene_cols = np.ones_like(scene_pts, dtype=np.float32) * 0.65
            scene_pts, scene_cols = self._prepare_scene_context(scene_pts, scene_cols)
            self.after_scene_pts, self.after_scene_cols = self._subsample_points_colors(
                scene_pts, scene_cols, self.SCENE_MAX_PTS
            )
        else:
            self.after_scene_pts = np.zeros((0, 3), dtype=np.float32)
            self.after_scene_cols = np.zeros((0, 3), dtype=np.float32)

        self.before_center = self._centroid(self.before_pts)
        self.after_center = self._centroid(self.after_pts)
        self.aligned_center = self._centroid(self.aligned_pts)

        self.yaw = 0.0
        self.pitch = 0.0
        self.zoom = 1.0
        self.drag_last = None
        self.show_mode = 0
        self.info_text = info_text
        self._standalone = parent is None

        if parent is not None:
            self.root = _tk.Toplevel(parent)
        else:
            self.root = _tk.Tk()
        self.root.title("In-Hand Pose Registration Viewer")
        self.root.geometry("1320x860")

        ctrl = _tk.Frame(self.root)
        ctrl.pack(side="top", fill="x", padx=8, pady=4)
        _tk.Label(ctrl, text="View:").pack(side="left")
        self.mode_var = _tk.StringVar(value="After Scene")
        modes = ["After Scene", "Object Overlap", "Motion Compare"]
        _tk.OptionMenu(ctrl, self.mode_var, *modes, command=self._on_mode_change).pack(side="left", padx=4)
        _tk.Label(ctrl, text="  drag rotate, wheel zoom").pack(side="left", padx=12)

        self.canvas = _tk.Canvas(self.root, bg="#0e141c", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _: self._render())
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<MouseWheel>", self._on_wheel)
        self.canvas.bind("<Button-4>", self._on_wheel)
        self.canvas.bind("<Button-5>", self._on_wheel)

        self.photo = None
        self.image_id = None

    @staticmethod
    def _subsample(pts: np.ndarray, max_pts: int) -> np.ndarray:
        if len(pts) <= max_pts:
            return pts
        idx = np.random.default_rng(42).choice(len(pts), max_pts, replace=False)
        return pts[idx]

    @staticmethod
    def _subsample_points_colors(
        pts: np.ndarray, cols: np.ndarray, max_pts: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(pts) <= max_pts:
            return pts, cols
        idx = np.random.default_rng(42).choice(len(pts), max_pts, replace=False)
        return pts[idx], cols[idx]

    @staticmethod
    def _stack_nonempty(*arrays: np.ndarray) -> np.ndarray:
        nonempty = [a for a in arrays if a is not None and len(a) > 0]
        if not nonempty:
            return np.zeros((0, 3), dtype=np.float32)
        return np.vstack(nonempty)

    @staticmethod
    def _centroid(pts: np.ndarray) -> np.ndarray:
        if len(pts) == 0:
            return np.zeros(3, dtype=np.float64)
        return np.median(pts.astype(np.float64), axis=0)

    @staticmethod
    def _diag(pts: np.ndarray) -> float:
        if len(pts) == 0:
            return 0.08
        return float(np.linalg.norm(np.ptp(pts.astype(np.float64), axis=0)))

    @staticmethod
    def _bgr_from_rgb(color_rgb) -> tuple[int, int, int]:
        return tuple(int(v) for v in (np.asarray(color_rgb)[::-1] * 255))

    def _prepare_scene_context(
        self, pts: np.ndarray, cols: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(pts) == 0:
            return pts, cols
        dist = np.linalg.norm(pts - self.object_center.reshape(1, 3), axis=1)
        radius = max(0.10, min(0.30, self.object_diag * 5.0))
        keep = dist <= radius
        if int(np.count_nonzero(keep)) < 4000:
            keep = dist <= min(0.38, radius * 1.8)
        if int(np.count_nonzero(keep)) > 0:
            pts = pts[keep]
            cols = cols[keep]

        gray = np.mean(cols, axis=1, keepdims=True)
        cols = np.clip(gray * 0.60 + cols * 0.40, 0.0, 1.0)
        cols = np.clip(cols * 0.40 + 0.08, 0.0, 1.0)
        return pts, cols

    def _on_mode_change(self, _=None):
        val = self.mode_var.get()
        if val == "After Scene":
            self.show_mode = 0
        elif val == "Object Overlap":
            self.show_mode = 1
        else:
            self.show_mode = 2
        self._render()

    def _on_press(self, event):
        self.drag_last = (event.x, event.y)

    def _on_drag(self, event):
        if self.drag_last is None:
            return
        dx = event.x - self.drag_last[0]
        dy = event.y - self.drag_last[1]
        self.yaw += dx * 0.4
        self.pitch += dy * 0.4
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self.drag_last = (event.x, event.y)
        self._render()

    def _on_release(self, _):
        self.drag_last = None

    def _on_wheel(self, event):
        delta = 0.0
        if hasattr(event, "delta") and event.delta:
            delta = event.delta / 120.0
        elif event.num == 4:
            delta = 1.0
        elif event.num == 5:
            delta = -1.0
        self.zoom *= 1.1 ** delta
        self.zoom = max(0.1, min(20.0, self.zoom))
        self._render()

    def _mode_payload(self):
        if self.show_mode == 0:
            focus = self._stack_nonempty(self.after_scene_pts, self.after_pts, self.aligned_pts)
            legend = [
                ("scene rgb", (145, 145, 145)),
                ("after object", self._bgr_from_rgb(self.AFTER_RGB)),
                ("aligned before", self._bgr_from_rgb(self.BEFORE_RGB)),
            ]
            title = "After scene reference"
            hint = "Dim RGB scene in back, orange target object, cyan aligned result."
            return focus, self.object_center, title, hint, legend
        if self.show_mode == 1:
            focus = self._stack_nonempty(self.after_pts, self.aligned_pts)
            legend = [
                ("after object", self._bgr_from_rgb(self.AFTER_RGB)),
                ("aligned before", self._bgr_from_rgb(self.BEFORE_RGB)),
            ]
            title = "Object overlap"
            hint = "Object-only view for checking shape overlap."
            return focus, self.object_center, title, hint, legend
        focus = self._stack_nonempty(self.after_scene_pts, self.before_pts, self.after_pts, self.aligned_pts)
        legend = [
            ("scene rgb", (145, 145, 145)),
            ("before raw", self._bgr_from_rgb(self.BEFORE_ORIG_RGB)),
            ("aligned before", self._bgr_from_rgb(self.BEFORE_RGB)),
            ("after object", self._bgr_from_rgb(self.AFTER_RGB)),
        ]
        title = "Motion compare"
        hint = "Green to cyan line shows how the before object moved."
        motion_target = self._centroid(self._stack_nonempty(self.before_pts, self.after_pts, self.aligned_pts))
        return focus, motion_target, title, hint, legend

    def _draw_cloud(self, img, pts, target, view_rot, scale, color_rgb, radius=1):
        if len(pts) == 0:
            return
        centered = pts.astype(np.float64) - target.reshape(1, 3)
        view = centered @ view_rot.T
        sx, sy, depth = view[:, 0], -view[:, 2], view[:, 1]
        cw, ch = img.shape[1], img.shape[0]
        xp = np.rint(cw * 0.5 + sx * scale).astype(np.int32)
        yp = np.rint(ch * 0.5 + sy * scale).astype(np.int32)
        valid = (xp >= 0) & (xp < cw) & (yp >= 0) & (yp < ch)
        xp, yp, depth = xp[valid], yp[valid], depth[valid]
        order = np.argsort(depth)
        xp, yp = xp[order], yp[order]
        color_bgr = tuple(int(v) for v in (np.asarray(color_rgb)[::-1] * 255))
        if radius <= 1:
            img[yp, xp] = np.array(color_bgr, dtype=np.uint8)
        else:
            for i in range(len(xp)):
                cv2.circle(img, (int(xp[i]), int(yp[i])), radius, color_bgr, -1, cv2.LINE_AA)

    def _draw_cloud_rgb(self, img, pts, cols_rgb, target, view_rot, scale, radius=1):
        if len(pts) == 0:
            return
        centered = pts.astype(np.float64) - target.reshape(1, 3)
        view = centered @ view_rot.T
        sx, sy, depth = view[:, 0], -view[:, 2], view[:, 1]
        cw, ch = img.shape[1], img.shape[0]
        xp = np.rint(cw * 0.5 + sx * scale).astype(np.int32)
        yp = np.rint(ch * 0.5 + sy * scale).astype(np.int32)
        valid = (xp >= 0) & (xp < cw) & (yp >= 0) & (yp < ch)
        xp, yp, depth = xp[valid], yp[valid], depth[valid]
        cols = np.clip(cols_rgb[valid][:, ::-1] * 255, 0, 255).astype(np.uint8)
        order = np.argsort(depth)
        xp, yp, cols = xp[order], yp[order], cols[order]
        if radius <= 1:
            img[yp, xp] = cols
        else:
            for i in range(len(xp)):
                cv2.circle(img, (int(xp[i]), int(yp[i])), radius,
                           tuple(int(v) for v in cols[i]), -1, cv2.LINE_AA)

    def _project_point(self, pt, target, view_rot, scale, cw, ch):
        view = (pt.astype(np.float64) - target.astype(np.float64)) @ view_rot.T
        return (
            int(round(cw * 0.5 + view[0] * scale)),
            int(round(ch * 0.5 - view[2] * scale)),
        )

    def _draw_segment(self, img, p0, p1, target, view_rot, scale, color_bgr):
        cw, ch = img.shape[1], img.shape[0]
        x0, y0 = self._project_point(p0, target, view_rot, scale, cw, ch)
        x1, y1 = self._project_point(p1, target, view_rot, scale, cw, ch)
        cv2.line(img, (x0, y0), (x1, y1), color_bgr, 2, cv2.LINE_AA)

    def _draw_legend(self, img, title, hint, legend):
        panel_w = min(img.shape[1] - 24, 500)
        panel_h = 78 + 22 * len(legend)
        overlay = img.copy()
        cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (18, 28, 38), -1)
        cv2.rectangle(overlay, (12, 12), (12 + panel_w, 12 + panel_h), (60, 82, 104), 1)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0.0, img)

        cv2.putText(img, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (238, 238, 238), 2, cv2.LINE_AA)
        cv2.putText(img, hint, (24, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (188, 188, 188), 1, cv2.LINE_AA)
        y = 88
        for text, color_bgr in legend:
            cv2.rectangle(img, (24, y - 11), (42, y + 5), color_bgr, -1)
            cv2.putText(img, text, (54, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
            y += 24

    def _render(self):
        cw = max(64, self.canvas.winfo_width())
        ch = max(64, self.canvas.winfo_height())

        import math as _math
        base_rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
        yaw_r = _math.radians(self.yaw)
        pitch_r = _math.radians(self.pitch)
        rz = np.array([
            [_math.cos(yaw_r), -_math.sin(yaw_r), 0],
            [_math.sin(yaw_r), _math.cos(yaw_r), 0],
            [0, 0, 1],
        ], dtype=np.float64)
        rx = np.array([
            [1, 0, 0],
            [0, _math.cos(pitch_r), -_math.sin(pitch_r)],
            [0, _math.sin(pitch_r), _math.cos(pitch_r)],
        ], dtype=np.float64)
        view_rot = (rx @ rz) @ base_rot

        focus_pts, target, title, hint, legend = self._mode_payload()
        if len(focus_pts) == 0:
            return

        centered = focus_pts.astype(np.float64) - target.reshape(1, 3)
        focus_view = centered @ view_rot.T
        px, py = focus_view[:, 0], -focus_view[:, 2]
        span_x = max(float(np.ptp(px)), 1e-3)
        span_y = max(float(np.ptp(py)), 1e-3)
        fit_scale = 0.72 * min((cw - 64) / span_x, (ch - 84) / span_y)
        scale = max(1e-3, fit_scale * self.zoom)

        img = np.zeros((ch, cw, 3), dtype=np.uint8)
        img[:] = self.BG_RGB

        if self.show_mode in (0, 2):
            self._draw_cloud_rgb(img, self.after_scene_pts, self.after_scene_cols, target, view_rot, scale, radius=1)

        if self.show_mode == 0:
            self._draw_cloud(img, self.after_pts, target, view_rot, scale, self.AFTER_RGB, radius=5)
            self._draw_cloud(img, self.aligned_pts, target, view_rot, scale, self.BEFORE_RGB, radius=5)
        elif self.show_mode == 1:
            self._draw_cloud(img, self.after_pts, target, view_rot, scale, self.AFTER_RGB, radius=5)
            self._draw_cloud(img, self.aligned_pts, target, view_rot, scale, self.BEFORE_RGB, radius=5)
        else:
            self._draw_cloud(img, self.before_pts, target, view_rot, scale, self.BEFORE_ORIG_RGB, radius=4)
            self._draw_cloud(img, self.after_pts, target, view_rot, scale, self.AFTER_RGB, radius=5)
            self._draw_cloud(img, self.aligned_pts, target, view_rot, scale, self.BEFORE_RGB, radius=5)
            self._draw_segment(img, self.before_center, self.aligned_center, target, view_rot, scale, self.TRAIL_RGB)

        self._draw_legend(img, title, hint, legend)

        bottom_lines = self.info_text.split("\n")[:8]
        for i, line in enumerate(bottom_lines):
            cv2.putText(
                img, line, (12, ch - 12 - (len(bottom_lines) - 1 - i) * 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA,
            )

        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.photo = self._ImageTk.PhotoImage(pil_img)
        if self.image_id is None:
            self.image_id = self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        else:
            self.canvas.itemconfig(self.image_id, image=self.photo)

    def run(self):
        self._render()
        if self._standalone:
            self.root.mainloop()


def run_local(
    before_left_path: str, before_right_path: str,
    after_left_path: str, after_right_path: str,
    intrinsic_path: str,
    object_name: str = "",
    sam3_url: str = DEFAULT_SAM3_URL,
    s2m2_url: str = DEFAULT_S2M2_URL,
    voxel_size: float = 0.003,
    use_coarse_fine: bool = True,
    save_dir: str | None = None,
    parent=None,
    show_viewer: bool = True,
    log_fn=None,
):
    """Run the full pipeline locally with tkinter viewer (same style as pointcloud_viewer.py)."""
    _log_fn = log_fn or (lambda msg: print(msg))
    ret: Dict[str, Any] = {
        "success": False,
        "result": None,
        "info_lines": [],
        "before_mask_vis": None,
        "after_mask_vis": None,
        "reg2d_vis": None,
        "before_shape_vis": None,
        "after_shape_vis": None,
        "before_det_debug": None,
        "after_det_debug": None,
        "detection_summary": "",
        "shape_summary": "",
        "save_dir": None,
        "pcds": None,
        "scene_after": None,
        "overlay_image_path": None,
        "bundle_path": None,
        "error": None,
        "error_traceback": None,
    }

    _log_fn("=" * 60)
    _log_fn("  Post-Grasp In-Hand Pose Re-Estimation (Local Mode)")
    _log_fn("=" * 60)

    before_left = cv2.cvtColor(cv2.imread(before_left_path), cv2.COLOR_BGR2RGB)
    before_right = cv2.cvtColor(cv2.imread(before_right_path), cv2.COLOR_BGR2RGB)
    after_left = cv2.cvtColor(cv2.imread(after_left_path), cv2.COLOR_BGR2RGB)
    after_right = cv2.cvtColor(cv2.imread(after_right_path), cv2.COLOR_BGR2RGB)
    _log_fn(f"  Images: before={before_left.shape}, after={after_left.shape}")

    intr = _parse_intrinsic_file(intrinsic_path)
    fx, fy, cx, cy, baseline = intr["fx"], intr["fy"], intr["cx"], intr["cy"], intr["baseline"]
    h, w = before_left.shape[:2]
    K = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)
    _log_fn(f"  Intrinsics: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f} baseline={baseline:.4f}m")

    _log_fn("\n[1/4] S2M2 depth estimation...")
    t0 = time.perf_counter()
    before_depth = depth_to_meters(call_s2m2_depth(before_left, before_right, fx, fy, cx, cy, baseline, s2m2_url), target_hw=(h, w))
    _log_fn(f"  Before: valid={int((before_depth > 0).sum())}, {int((time.perf_counter()-t0)*1000)}ms")
    t0 = time.perf_counter()
    ah, aw = after_left.shape[:2]
    after_depth = depth_to_meters(call_s2m2_depth(after_left, after_right, fx, fy, cx, cy, baseline, s2m2_url), target_hw=(ah, aw))
    _log_fn(f"  After:  valid={int((after_depth > 0).sum())}, {int((time.perf_counter()-t0)*1000)}ms")

    _log_fn("\n[2/4] Qwen detection + SAM3 segmentation...")
    obj_prompt = object_name.strip() or None
    t0 = time.perf_counter()
    before_mask, before_bbox, before_det_debug = detect_and_segment(before_left, text_prompt=obj_prompt, sam3_url=sam3_url)
    _log_fn(f"  Before mask: {int((before_mask > 0).sum())}px, bbox={before_bbox}, {int((time.perf_counter()-t0)*1000)}ms")
    t0 = time.perf_counter()
    after_mask, after_bbox, after_det_debug = detect_and_segment(after_left, text_prompt=obj_prompt, sam3_url=sam3_url)
    _log_fn(f"  After mask:  {int((after_mask > 0).sum())}px, bbox={after_bbox}, {int((time.perf_counter()-t0)*1000)}ms")
    before_overlap = int(((before_mask > 0) & (before_depth > 0)).sum())
    after_overlap = int(((after_mask > 0) & (after_depth > 0)).sum())
    _log_fn(f"  Before mask∩depth: {before_overlap}")
    _log_fn(f"  After  mask∩depth: {after_overlap}")

    before_mask_vis = _draw_detection_overlay(
        before_left, before_mask, before_bbox,
        color=(0, 180, 0), box_color=(0, 255, 0),
        label=before_det_debug.get("source", ""),
    )
    after_mask_vis = _draw_detection_overlay(
        after_left, after_mask, after_bbox,
        color=(0, 140, 190), box_color=(0, 128, 255),
        label=after_det_debug.get("source", ""),
    )
    ret["before_mask_vis"] = before_mask_vis
    ret["after_mask_vis"] = after_mask_vis
    ret["before_det_debug"] = before_det_debug
    ret["after_det_debug"] = after_det_debug
    ret["detection_summary"] = "\n\n".join([
        _build_detection_debug_text("Before", before_det_debug, before_bbox, before_mask, before_depth),
        _build_detection_debug_text("After", after_det_debug, after_bbox, after_mask, after_depth),
    ])

    if before_overlap < 80 or after_overlap < 80:
        _log_fn("  Warning: mask∩depth 很小，Qwen/SAM3 结果可能不稳定，下面会尝试宽松点云过滤重试。")

    _log_fn("\n[3/4] Point cloud registration...")
    cfg = RegistrationConfig(voxel_size=voxel_size)
    cfg_used = cfg
    try:
        result = run_reestimation(
            before_rgb=before_left, before_depth=before_depth, before_mask=before_mask,
            after_rgb=after_left, after_depth=after_depth, after_mask=after_mask,
            K_wrist=K, use_coarse_fine=use_coarse_fine, cfg=cfg_used,
        )
    except Exception as first_err:
        err_msg = str(first_err)
        if (
            "No valid object points after mask/depth filtering" in err_msg
            or "Point cloud became empty" in err_msg
        ):
            cfg_used = replace(
                cfg,
                erode_kernel=1,
                erode_iters=0,
                remove_depth_edge_points=False,
                iqr_multiplier=0.0,
                outlier_nb_neighbors=8,
                outlier_std_ratio=2.5,
                radius_outlier_radius=0.0,
                radius_outlier_min_neighbors=0,
                min_visible_points=20,
            )
            _log_fn("  初次点云构建失败，使用宽松过滤参数重试...")
            try:
                result = run_reestimation(
                    before_rgb=before_left, before_depth=before_depth, before_mask=before_mask,
                    after_rgb=after_left, after_depth=after_depth, after_mask=after_mask,
                    K_wrist=K, use_coarse_fine=use_coarse_fine, cfg=cfg_used,
                )
            except Exception as retry_err:
                ret["error"] = str(retry_err)
                ret["error_traceback"] = traceback.format_exc()
                _log_fn(f"\n{'='*50}\n失败: {ret['error']}\n{ret['error_traceback']}")
                return ret
        else:
            ret["error"] = err_msg
            ret["error_traceback"] = traceback.format_exc()
            _log_fn(f"\n{'='*50}\n失败: {err_msg}\n{ret['error_traceback']}")
            return ret

    q = result.debug.get("quality", {})
    t_vec = result.T_slip[:3, 3]
    frame_name = result.debug.get("t_slip_frame", "camera")
    info_lines = [
        f"Quality: {q.get('quality', '?').upper()} | Mode: {result.mode} | {result.elapsed_s:.2f}s",
        f"Init: {result.debug.get('best_init', '?')}",
        f"GICP fitness: {result.debug.get('reg_gicp_fitness', '?')} | RMSE: {result.debug.get('reg_gicp_inlier_rmse', '?')}",
        f"T_slip trans: [{t_vec[0]:.4f}, {t_vec[1]:.4f}, {t_vec[2]:.4f}]m ({np.linalg.norm(t_vec):.4f}m)",
    ]
    if result.debug.get("reg_axis_prior_weight", 0.0) > 1e-6:
        info_lines.append(
            "Elongated prior:"
            f" w={result.debug.get('reg_axis_prior_weight', 0.0):.3f}"
            f" axis={result.debug.get('reg_axis_alignment_score', 0.0):.3f}"
            f" center={result.debug.get('reg_axis_center_score', 0.0):.3f}"
        )
    if result.debug.get("reg_silhouette_f1") is not None:
        info_lines.append(
            "Proj mask overlap:"
            f" F1={result.debug.get('reg_silhouette_f1', 0.0):.3f}"
            f" IoU={result.debug.get('reg_silhouette_iou', 0.0):.3f}"
            f" P={result.debug.get('reg_silhouette_precision', 0.0):.3f}"
            f" R={result.debug.get('reg_silhouette_recall', 0.0):.3f}"
        )
    if result.debug.get("reg_cloud_overlap_enabled"):
        info_lines.append(
            "Cloud overlap:"
            f" F1={result.debug.get('reg_cloud_overlap_f1', 0.0):.3f}"
            f" P={result.debug.get('reg_cloud_overlap_precision', 0.0):.3f}"
            f" R={result.debug.get('reg_cloud_overlap_recall', 0.0):.3f}"
            f" chamfer={1000.0 * result.debug.get('reg_cloud_overlap_chamfer_m', 0.0):.1f}mm"
        )
    if result.debug.get("reg_shape_weight", 0.0) > 1e-6:
        info_lines.append(
            "Shape prior:"
            f" w={result.debug.get('reg_shape_weight', 0.0):.3f}"
            f" score={result.debug.get('reg_shape_score', 0.0):.3f}"
            f" center={result.debug.get('reg_shape_center_score', 0.0):.3f}"
            f" axis={result.debug.get('reg_shape_axis_score', 0.0):.3f}"
        )
    shape_pair_info = result.debug.get("shape_prior_pair") or {}
    if shape_pair_info.get("selected"):
        info_lines.append(
            "Shape pair:"
            f" src={shape_pair_info.get('source_type')}"
            f" tgt={shape_pair_info.get('target_type')}"
            f" mode={shape_pair_info.get('selection_mode')}"
            f" compat={shape_pair_info.get('compatibility', 0.0):.3f}"
            f" conf={shape_pair_info.get('confidence', 0.0):.3f}"
        )
        if (
            shape_pair_info.get("source_type") != shape_pair_info.get("source_best_type")
            or shape_pair_info.get("target_type") != shape_pair_info.get("target_best_type")
        ):
            info_lines.append(
                "Shape pair fallback:"
                f" best=({shape_pair_info.get('source_best_type')}->{shape_pair_info.get('target_best_type')})"
                f" selected=({shape_pair_info.get('source_type')}->{shape_pair_info.get('target_type')})"
            )
    relation_info = result.debug.get("shape_relation_info") or {}
    if relation_info.get("available"):
        info_lines.append(
            "M3 endpoint cue:"
            f" pref={relation_info.get('preferred_relation')}"
            f" conf={relation_info.get('confidence', 0.0):.3f}"
            f" same={relation_info.get('same_score', 0.0):.3f}"
            f" reversed={relation_info.get('reversed_score', 0.0):.3f}"
        )
    if result.debug.get("reg_m3_flip_applied"):
        info_lines.append("M3 flip correction: applied")
    if result.debug.get("reg_shape_snap_applied"):
        info_lines.append(f"Shape center snap: {result.debug.get('reg_shape_snap_applied')}")
    if result.debug.get("reg_micro_refine_applied"):
        info_lines.append(f"Metric micro-refine: {result.debug.get('reg_micro_refine_applied')}")
    if cfg_used is not cfg:
        info_lines.append("Point-cloud filtering: relaxed retry was used")
    info_lines.extend(_t_slip_rotation_lines(result.T_slip, frame_name, compact=True))

    shape_summary_lines: list[str] = []
    shape_pair_info = result.debug.get("shape_prior_pair") or {}
    if shape_pair_info.get("selected"):
        shape_summary_lines.append(
            "Prior pair:"
            f" src={shape_pair_info.get('source_type')}"
            f" tgt={shape_pair_info.get('target_type')}"
            f" mode={shape_pair_info.get('selection_mode')}"
            f" compat={shape_pair_info.get('compatibility', 0.0):.3f}"
            f" conf={shape_pair_info.get('confidence', 0.0):.3f}"
        )
        if (
            shape_pair_info.get("source_type") != shape_pair_info.get("source_best_type")
            or shape_pair_info.get("target_type") != shape_pair_info.get("target_best_type")
        ):
            shape_summary_lines.append(
                "Prior pair fallback:"
                f" best=({shape_pair_info.get('source_best_type')}->{shape_pair_info.get('target_best_type')})"
                f" selected=({shape_pair_info.get('source_type')}->{shape_pair_info.get('target_type')})"
            )
        shape_summary_lines.append("")
    if result.debug.get("shape_before_fit"):
        shape_summary_lines.extend(summarize_fit_result_lines(result.debug.get("shape_before_fit"), prefix="Before"))
    if result.debug.get("shape_after_fit"):
        if shape_summary_lines:
            shape_summary_lines.append("")
        shape_summary_lines.extend(summarize_fit_result_lines(result.debug.get("shape_after_fit"), prefix="After"))
    if result.debug.get("shape_before_fit_error"):
        shape_summary_lines.append(f"Before fit error: {result.debug.get('shape_before_fit_error')}")
    if result.debug.get("shape_after_fit_error"):
        shape_summary_lines.append(f"After fit error: {result.debug.get('shape_after_fit_error')}")
    ret["shape_summary"] = "\n".join(shape_summary_lines)

    for line in info_lines:
        _log_fn(f"  {line}")
    _log_fn(f"\n  T_registration:\n{np.array2string(result.T_registration, precision=6, suppress_small=True)}")

    _log_fn("\n[4/4] Building point clouds...")
    try:
        bp = rgbd_to_pointcloud(before_left, before_depth, before_mask, K, cfg_used)
        bp = preprocess_pcd(bp, cfg_used)
        ap = rgbd_to_pointcloud(after_left, after_depth, after_mask, K, cfg_used)
        ap = preprocess_pcd(ap, cfg_used)
        aligned = copy_pcd(bp)
        aligned.transform(result.T_registration)
        after_scene_pcd = _build_context_scene_pcd(
            after_left, after_depth, after_mask, K,
            voxel_size=max(voxel_size * 1.5, 0.006),
        )
        reg2d_vis = _draw_registration_overlay_2d(
            after_left, after_mask, after_depth, K,
            after_pcd=ap, aligned_pcd=aligned,
        )

        before_shape_raw = result.debug.get("_shape_before_fit_raw")
        after_shape_raw = result.debug.get("_shape_after_fit_raw")
        K_mat = np.array([
            [K.fx, 0.0, K.cx],
            [0.0, K.fy, K.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        if before_shape_raw:
            before_shape_vis = render_best_fit_overlay_image(
                image_bgr=cv2.cvtColor(before_left, cv2.COLOR_RGB2BGR),
                object_points_cam=np.asarray(bp.points, dtype=np.float64),
                fit_result=before_shape_raw,
                K=K_mat,
            )
            ret["before_shape_vis"] = cv2.cvtColor(before_shape_vis, cv2.COLOR_BGR2RGB)
        if after_shape_raw:
            after_shape_vis = render_best_fit_overlay_image(
                image_bgr=cv2.cvtColor(after_left, cv2.COLOR_RGB2BGR),
                object_points_cam=np.asarray(ap.points, dtype=np.float64),
                fit_result=after_shape_raw,
                K=K_mat,
            )
            ret["after_shape_vis"] = cv2.cvtColor(after_shape_vis, cv2.COLOR_BGR2RGB)
    except Exception as build_err:
        ret["result"] = result
        ret["info_lines"] = info_lines
        ret["error"] = str(build_err)
        ret["error_traceback"] = traceback.format_exc()
        _log_fn(f"\n{'='*50}\n失败: {build_err}\n{ret['error_traceback']}")
        return ret

    if save_dir is None:
        save_dir = tempfile.mkdtemp(prefix="pose_inhand_local_")
    os.makedirs(save_dir, exist_ok=True)
    o3d.io.write_point_cloud(os.path.join(save_dir, "object_before.ply"), bp)
    o3d.io.write_point_cloud(os.path.join(save_dir, "object_after.ply"), ap)
    o3d.io.write_point_cloud(os.path.join(save_dir, "object_before_aligned.ply"), aligned)
    overlay_path = os.path.join(save_dir, "overlay_after_registration.png")
    cv2.imwrite(overlay_path, cv2.cvtColor(reg2d_vis, cv2.COLOR_RGB2BGR))
    before_shape_overlay_path = None
    after_shape_overlay_path = None
    if ret.get("before_shape_vis") is not None:
        before_shape_overlay_path = os.path.join(save_dir, "before_shape_overlay.png")
        cv2.imwrite(before_shape_overlay_path, cv2.cvtColor(ret["before_shape_vis"], cv2.COLOR_RGB2BGR))
    if ret.get("after_shape_vis") is not None:
        after_shape_overlay_path = os.path.join(save_dir, "after_shape_overlay.png")
        cv2.imwrite(after_shape_overlay_path, cv2.cvtColor(ret["after_shape_vis"], cv2.COLOR_RGB2BGR))
    if result.debug.get("shape_before_fit"):
        with open(os.path.join(save_dir, "shape_before_fit.json"), "w") as f:
            json.dump(result.debug.get("shape_before_fit"), f, indent=2, default=str)
    if result.debug.get("shape_after_fit"):
        with open(os.path.join(save_dir, "shape_after_fit.json"), "w") as f:
            json.dump(result.debug.get("shape_after_fit"), f, indent=2, default=str)
    if ret.get("shape_summary"):
        with open(os.path.join(save_dir, "shape_fit_summary.txt"), "w") as f:
            f.write(ret["shape_summary"] + "\n")
    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump({"success": result.success, "mode": result.mode,
                    "T_slip": result.T_slip.tolist(), "T_registration": result.T_registration.tolist(),
                    "quality": q, "elapsed_s": result.elapsed_s,
                    "best_init": result.debug.get("best_init"),
                    "shape_init_candidates": result.debug.get("shape_init_candidates"),
                    "shape_before_confidence": result.debug.get("shape_before_confidence"),
                    "shape_after_confidence": result.debug.get("shape_after_confidence"),
                    "axis_prior_weight": result.debug.get("reg_axis_prior_weight"),
                    "axis_alignment_score": result.debug.get("reg_axis_alignment_score"),
                    "axis_center_score": result.debug.get("reg_axis_center_score"),
                    "axis_length_score": result.debug.get("reg_axis_length_score"),
                    "silhouette_f1": result.debug.get("reg_silhouette_f1"),
                    "silhouette_iou": result.debug.get("reg_silhouette_iou"),
                    "silhouette_precision": result.debug.get("reg_silhouette_precision"),
                    "silhouette_recall": result.debug.get("reg_silhouette_recall"),
                    "cloud_overlap_enabled": result.debug.get("reg_cloud_overlap_enabled"),
                    "cloud_overlap_weight": result.debug.get("reg_cloud_overlap_weight"),
                    "cloud_overlap_score": result.debug.get("reg_cloud_overlap_score"),
                    "cloud_overlap_f1": result.debug.get("reg_cloud_overlap_f1"),
                    "cloud_overlap_precision": result.debug.get("reg_cloud_overlap_precision"),
                    "cloud_overlap_recall": result.debug.get("reg_cloud_overlap_recall"),
                    "cloud_overlap_chamfer_m": result.debug.get("reg_cloud_overlap_chamfer_m"),
                    "cloud_overlap_robust_m": result.debug.get("reg_cloud_overlap_robust_m"),
                    "shape_weight": result.debug.get("reg_shape_weight"),
                    "shape_score": result.debug.get("reg_shape_score"),
                    "shape_center_score": result.debug.get("reg_shape_center_score"),
                    "shape_axis_score": result.debug.get("reg_shape_axis_score"),
                    "shape_size_score": result.debug.get("reg_shape_size_score"),
                    "shape_endpoint_score": result.debug.get("reg_shape_endpoint_score"),
                    "shape_rotation_score": result.debug.get("reg_shape_rotation_score"),
                    "shape_relation_score": result.debug.get("reg_shape_relation_score"),
                    "shape_prior_pair": result.debug.get("shape_prior_pair"),
                    "shape_relation_info": result.debug.get("shape_relation_info"),
                    "m3_flip_applied": result.debug.get("reg_m3_flip_applied"),
                    "shape_snap_applied": result.debug.get("reg_shape_snap_applied"),
                    "micro_refine_applied": result.debug.get("reg_micro_refine_applied"),
                    "micro_refine_sequence": result.debug.get("micro_refine_sequence"),
                    "shape_before_fit": result.debug.get("shape_before_fit"),
                    "shape_after_fit": result.debug.get("shape_after_fit")}, f, indent=2, default=str)
    bundle_path = os.path.join(save_dir, "result_bundle.zip")
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_name in [
            "object_before.ply",
            "object_after.ply",
            "object_before_aligned.ply",
            "overlay_after_registration.png",
            "result.json",
            "before_shape_overlay.png",
            "after_shape_overlay.png",
            "shape_before_fit.json",
            "shape_after_fit.json",
            "shape_fit_summary.txt",
        ]:
            file_path = os.path.join(save_dir, file_name)
            if os.path.isfile(file_path):
                zf.write(file_path, file_name)
    _log_fn(f"  Saved to: {save_dir}")

    if show_viewer:
        _log_fn("\n  Opening viewer (drag=rotate, scroll=zoom, dropdown=switch view)...\n")
        viewer = _LocalResultViewer(
            bp, ap, aligned,
            after_scene_pcd=after_scene_pcd,
            info_text="\n".join(info_lines),
            parent=parent,
        )
        viewer.run()

    ret["success"] = True
    ret["result"] = result
    ret["info_lines"] = info_lines
    ret["save_dir"] = save_dir
    ret["scene_after"] = after_scene_pcd
    ret["reg2d_vis"] = reg2d_vis
    ret["overlay_image_path"] = overlay_path
    ret["bundle_path"] = bundle_path
    ret["pcds"] = {"before": bp, "after": ap, "aligned": aligned, "after_scene": after_scene_pcd}
    return ret


# ============================================================================
# StereoNet result browser — browse & pick before/after from data dir
# ============================================================================

class _StereoNetBrowser:
    """Browse stereonet_result data and run pipeline on selected before/after pairs.

    Layout: left = input (index list + image selection), right = output (results).
    """

    def __init__(
        self, data_dir: str,
        sam3_url: str = DEFAULT_SAM3_URL,
        s2m2_url: str = DEFAULT_S2M2_URL,
        voxel_size: float = 0.003,
        use_coarse_fine: bool = True,
        object_name: str = "",
    ):
        import tkinter as _tk
        from tkinter import ttk as _ttk
        from PIL import Image as PILImage, ImageTk as _ImageTk

        self._tk = _tk
        self._ttk = _ttk
        self._PILImage = PILImage
        self._ImageTk = _ImageTk
        self.data_dir = os.path.abspath(data_dir)
        self.sam3_url = sam3_url
        self.s2m2_url = s2m2_url
        self.voxel_size = voxel_size
        self.use_coarse_fine = use_coarse_fine

        self.indices = self._scan_indices()
        if not self.indices:
            raise RuntimeError(f"No origin_L/R data found in {data_dir}")

        intr_path = os.path.join(self.data_dir, "camera_intrinsic_origin.txt")
        if not os.path.isfile(intr_path):
            raise RuntimeError(f"Missing {intr_path}")
        self.intrinsic_path = intr_path

        self.before_idx: str | None = self.indices[0] if self.indices else None
        self.after_idx: str | None = self.indices[1] if len(self.indices) > 1 else None
        self._last_run_pcds = None
        self._last_run_info = ""
        self._download_path: str | None = None
        self._result_dir_path: str | None = None

        self.root = _tk.Tk()
        self.root.title("Local App — StereoNet Browser")
        self.root.geometry("1600x940")
        self.obj_name_var = _tk.StringVar(value=object_name)
        self._build_ui()

    def _scan_indices(self) -> list[str]:
        indices: set[str] = set()
        for f in os.listdir(self.data_dir):
            if f.endswith("_origin_L.png"):
                idx_str = f[: -len("_origin_L.png")]
                if os.path.isfile(os.path.join(self.data_dir, f"{idx_str}_origin_R.png")):
                    indices.add(idx_str)
        return sorted(indices)

    def _build_ui(self):
        _tk, _ttk = self._tk, self._ttk

        # ── Title bar ──
        title_frame = _ttk.Frame(self.root, padding=(8, 4))
        title_frame.pack(fill="x")
        _ttk.Label(
            title_frame,
            text=f"StereoNet Browser — {len(self.indices)} 组数据",
            font=("", 14, "bold"),
        ).pack(side="left")
        _ttk.Label(
            title_frame,
            text=f"路径: {self.data_dir}  |  内参: camera_intrinsic_origin.txt",
            foreground="gray",
        ).pack(side="right")

        # ── Main body: Left | Right ──
        body = _ttk.PanedWindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.body_pane = body

        # =====================================================================
        # LEFT PANEL — Input
        # =====================================================================
        left = _ttk.Frame(body)
        body.add(left, weight=1)

        # -- Index list + preview --
        top_left = _ttk.Frame(left)
        top_left.pack(fill="both", expand=True)

        # Index list
        list_frame = _ttk.LabelFrame(top_left, text="序号列表")
        list_frame.pack(side="left", fill="y", padx=(0, 4))
        list_inner = _ttk.Frame(list_frame)
        list_inner.pack(fill="both", expand=True)
        scrollbar = _ttk.Scrollbar(list_inner)
        scrollbar.pack(side="right", fill="y")
        self.listbox = _tk.Listbox(
            list_inner, width=10, yscrollcommand=scrollbar.set, font=("Courier", 11),
        )
        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.listbox.yview)
        for idx in self.indices:
            self.listbox.insert("end", idx)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        # Preview + buttons
        preview_panel = _ttk.Frame(top_left)
        preview_panel.pack(side="left", fill="both", expand=True)

        preview_lf = _ttk.LabelFrame(preview_panel, text="预览 (点击序号)")
        preview_lf.pack(fill="both", expand=True, pady=(0, 4))
        self.preview_canvas = _tk.Canvas(preview_lf, height=180, bg="#fafafa")
        self.preview_canvas.pack(fill="both", expand=True, padx=4, pady=4)
        self.preview_info_var = _tk.StringVar(value="当前高亮: -")
        _ttk.Label(
            preview_panel,
            textvariable=self.preview_info_var,
            foreground="gray",
        ).pack(fill="x", pady=(0, 2))

        btn_frame = _ttk.Frame(preview_panel)
        btn_frame.pack(fill="x", pady=4)
        _ttk.Button(btn_frame, text="设为 Before (抓取前)",
                     command=self._set_before).pack(side="left", padx=(0, 6))
        _ttk.Button(btn_frame, text="设为 After (抓取后)",
                     command=self._set_after).pack(side="left")

        # -- Before / After images side-by-side --
        pair_outer = _ttk.LabelFrame(left, text="当前选择")
        pair_outer.pack(fill="x", pady=(4, 0))

        self.before_var = _tk.StringVar(value="Before: 未选择")
        self.after_var = _tk.StringVar(value="After: 未选择")
        self.selection_summary_var = _tk.StringVar(value="当前配对: -")

        _ttk.Label(
            pair_outer,
            textvariable=self.selection_summary_var,
            foreground="gray",
        ).pack(anchor="w", padx=4, pady=(4, 0))

        pair = _ttk.Frame(pair_outer)
        pair.pack(fill="x", padx=4, pady=4)

        bf = _ttk.Frame(pair)
        bf.pack(side="left", fill="both", expand=True, padx=(0, 4))
        _ttk.Label(bf, textvariable=self.before_var, foreground="#2e7d32",
                    font=("", 10, "bold")).pack(anchor="w")
        self.before_canvas = _tk.Canvas(bf, height=140, bg="#e8f5e9")
        self.before_canvas.pack(fill="both", expand=True)

        af = _ttk.Frame(pair)
        af.pack(side="left", fill="both", expand=True, padx=(4, 0))
        _ttk.Label(af, textvariable=self.after_var, foreground="#1565c0",
                    font=("", 10, "bold")).pack(anchor="w")
        self.after_canvas = _tk.Canvas(af, height=140, bg="#e3f2fd")
        self.after_canvas.pack(fill="both", expand=True)

        # -- Run controls --
        run_frame = _ttk.Frame(left)
        run_frame.pack(fill="x", pady=(6, 0))
        _ttk.Label(run_frame, text="物体名称:").pack(side="left")
        _ttk.Entry(run_frame, textvariable=self.obj_name_var, width=14).pack(
            side="left", padx=(4, 8))
        _ttk.Button(run_frame, text="运行 Re-Estimation",
                     command=self._on_run).pack(side="left")
        self.run_status_var = _tk.StringVar(value="就绪")
        _ttk.Label(run_frame, textvariable=self.run_status_var,
                    foreground="gray").pack(side="left", padx=8)

        # =====================================================================
        # RIGHT PANEL — Output
        # =====================================================================
        right = _ttk.Frame(body)
        body.add(right, weight=4)
        right_outer = _ttk.Frame(right)
        right_outer.pack(fill="both", expand=True)
        self.results_scroll = _ttk.Scrollbar(right_outer, orient="vertical")
        self.results_scroll.pack(side="right", fill="y")
        self.results_canvas = _tk.Canvas(
            right_outer,
            bg=self.root.cget("bg"),
            highlightthickness=0,
            yscrollcommand=self.results_scroll.set,
        )
        self.results_canvas.pack(side="left", fill="both", expand=True)
        self.results_scroll.config(command=self.results_canvas.yview)

        right_lf = _ttk.LabelFrame(self.results_canvas, text="结果输出")
        self.results_inner = right_lf
        self.results_window = self.results_canvas.create_window((0, 0), window=right_lf, anchor="nw")
        self.results_canvas.bind("<Configure>", self._on_results_canvas_configure)
        right_lf.bind("<Configure>", self._on_results_inner_configure)

        action_row = _ttk.Frame(right_lf)
        action_row.pack(fill="x", padx=4, pady=(4, 2))
        self.download_btn = _ttk.Button(
            action_row, text="下载结果包", command=self._open_download_path, state="disabled",
        )
        self.download_btn.pack(side="left")
        self.result_dir_btn = _ttk.Button(
            action_row, text="打开结果目录", command=self._open_result_dir, state="disabled",
        )
        self.result_dir_btn.pack(side="left", padx=(6, 0))
        self.view3d_btn = _ttk.Button(
            action_row, text="打开 3D 点云查看器",
            command=self._open_3d_viewer, state="disabled",
        )
        self.view3d_btn.pack(side="left", padx=(6, 0))
        self.jump_2d_btn = _ttk.Button(
            action_row, text="跳到二维结果",
            command=self._scroll_to_reg2d,
        )
        self.jump_2d_btn.pack(side="left", padx=(6, 0))
        self.action_hint_var = _tk.StringVar(value="运行完成后，这里会提供下载、目录和 3D 查看入口")
        _ttk.Label(
            action_row, textvariable=self.action_hint_var, foreground="gray",
        ).pack(side="left", padx=10)

        # -- Result summary text --
        text_frame = _ttk.Frame(right_lf)
        text_frame.pack(fill="x", expand=False, padx=4, pady=(4, 2))
        text_scroll = _ttk.Scrollbar(text_frame)
        text_scroll.pack(side="right", fill="y")
        self.result_text = _tk.Text(
            text_frame, height=13, wrap="word", font=("SF Mono", 9),
            bg="#1e1e2e", fg="#cdd6f4", insertbackground="#cdd6f4",
            state="disabled", yscrollcommand=text_scroll.set,
        )
        self.result_text.pack(fill="x", expand=True)
        text_scroll.config(command=self.result_text.yview)
        self._log_to_text("等待运行...\n\n提示: 在左侧选择 Before 和 After 序号，然后点击 [运行]")

        # -- Detection / mask result images --
        mask_frame = _ttk.LabelFrame(right_lf, text="Qwen + SAM3 中间结果")
        mask_frame.pack(fill="both", expand=True, padx=4, pady=(2, 4))
        mask_inner = _ttk.Frame(mask_frame)
        mask_inner.pack(fill="x", padx=4, pady=4)

        bm_frame = _ttk.Frame(mask_inner)
        bm_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        _ttk.Label(bm_frame, text="Before: bbox + mask overlay").pack(anchor="w")
        self.before_mask_canvas = _tk.Canvas(bm_frame, height=190, bg="#f5f5f5")
        self.before_mask_canvas.pack(fill="both", expand=True)

        am_frame = _ttk.Frame(mask_inner)
        am_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
        _ttk.Label(am_frame, text="After: bbox + mask overlay").pack(anchor="w")
        self.after_mask_canvas = _tk.Canvas(am_frame, height=190, bg="#f5f5f5")
        self.after_mask_canvas.pack(fill="both", expand=True)

        det_text_frame = _ttk.Frame(mask_frame)
        det_text_frame.pack(fill="x", padx=4, pady=(0, 4))
        det_scroll = _ttk.Scrollbar(det_text_frame)
        det_scroll.pack(side="right", fill="y")
        self.detection_text = _tk.Text(
            det_text_frame, height=6, wrap="word", font=("SF Mono", 9),
            bg="#fafafa", fg="#303030", state="disabled",
            yscrollcommand=det_scroll.set,
        )
        self.detection_text.pack(fill="x", expand=True)
        det_scroll.config(command=self.detection_text.yview)
        _ttk.Label(mask_frame, text="提示: 双击任意结果图可放大查看", foreground="gray").pack(anchor="e", padx=6, pady=(0, 4))

        shape_frame = _ttk.LabelFrame(right_lf, text="Shape Fit 辅助结果")
        shape_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        shape_inner = _ttk.Frame(shape_frame)
        shape_inner.pack(fill="x", padx=4, pady=4)

        bs_frame = _ttk.Frame(shape_inner)
        bs_frame.pack(side="left", fill="both", expand=True, padx=(0, 4))
        _ttk.Label(bs_frame, text="Before: primitive fit overlay").pack(anchor="w")
        self.before_shape_canvas = _tk.Canvas(bs_frame, height=210, bg="#f5f5f5")
        self.before_shape_canvas.pack(fill="both", expand=True)

        as_frame = _ttk.Frame(shape_inner)
        as_frame.pack(side="left", fill="both", expand=True, padx=(4, 0))
        _ttk.Label(as_frame, text="After: primitive fit overlay").pack(anchor="w")
        self.after_shape_canvas = _tk.Canvas(as_frame, height=210, bg="#f5f5f5")
        self.after_shape_canvas.pack(fill="both", expand=True)

        shape_text_frame = _ttk.Frame(shape_frame)
        shape_text_frame.pack(fill="x", padx=4, pady=(0, 4))
        shape_scroll = _ttk.Scrollbar(shape_text_frame)
        shape_scroll.pack(side="right", fill="y")
        self.shape_text = _tk.Text(
            shape_text_frame, height=5, wrap="word", font=("SF Mono", 9),
            bg="#fafafa", fg="#303030", state="disabled",
            yscrollcommand=shape_scroll.set,
        )
        self.shape_text.pack(fill="x", expand=True)
        shape_scroll.config(command=self.shape_text.yview)

        reg2d_frame = _ttk.LabelFrame(right_lf, text="二维配准叠加")
        self.reg2d_frame = reg2d_frame
        reg2d_frame.pack(fill="both", expand=True, padx=4, pady=(0, 4))
        self.reg2d_canvas = _tk.Canvas(reg2d_frame, height=360, bg="#f5f5f5")
        self.reg2d_canvas.pack(fill="both", expand=True, padx=4, pady=(4, 2))

        link_row = _ttk.Frame(reg2d_frame)
        link_row.pack(fill="x", padx=4, pady=(0, 4))
        self.download_var = _tk.StringVar(value="下载包: 暂无")
        self.download_link = _tk.Label(
            link_row,
            textvariable=self.download_var,
            fg="#555555",
            cursor="arrow",
            anchor="w",
        )
        self.download_link.pack(side="left", fill="x", expand=True)
        _ttk.Label(link_row, text="右侧支持整体滚动；双击图片可放大", foreground="gray").pack(side="right")

        self._set_shape_text("等待运行后显示 primitive fit 类型、尺寸、残差和端点信息")
        self.root.after(80, self._set_initial_split)
        self._init_default_selection()

    # ── helpers ──

    def _set_initial_split(self):
        try:
            total_w = max(self.body_pane.winfo_width(), self.root.winfo_width() - 24, 1000)
            left_w = max(320, int(total_w * 0.28))
            self.body_pane.sashpos(0, left_w)
        except Exception:
            pass

    def _init_default_selection(self):
        if not self.indices:
            return
        self.listbox.selection_clear(0, "end")
        self.listbox.selection_set(0)
        self.listbox.activate(0)
        self.listbox.see(0)
        self._on_list_select()
        if self.before_idx is not None:
            self._refresh_before_preview()
        if self.after_idx is not None:
            self._refresh_after_preview()
        self._refresh_selection_summary()

    def _log_to_text(self, msg: str, append: bool = False):
        self.result_text.config(state="normal")
        if not append:
            self.result_text.delete("1.0", "end")
        self.result_text.insert("end", msg + "\n")
        self.result_text.see("end")
        self.result_text.config(state="disabled")
        self.root.update_idletasks()

    def _set_detection_text(self, msg: str):
        self.detection_text.config(state="normal")
        self.detection_text.delete("1.0", "end")
        self.detection_text.insert("end", msg)
        self.detection_text.config(state="disabled")
        self.root.update_idletasks()

    def _set_shape_text(self, msg: str):
        self.shape_text.config(state="normal")
        self.shape_text.delete("1.0", "end")
        self.shape_text.insert("end", msg)
        self.shape_text.config(state="disabled")
        self.root.update_idletasks()

    def _on_results_inner_configure(self, _event=None):
        if hasattr(self, "results_canvas"):
            self.results_canvas.configure(scrollregion=self.results_canvas.bbox("all"))

    def _on_results_canvas_configure(self, event):
        if hasattr(self, "results_window"):
            self.results_canvas.itemconfigure(self.results_window, width=event.width)
        self._on_results_inner_configure()

    def _scroll_results_to_widget(self, widget, padding: int = 8):
        if not hasattr(self, "results_canvas") or widget is None:
            return
        self.root.update_idletasks()
        bbox = self.results_canvas.bbox("all")
        if not bbox:
            return
        inner_h = max(int(bbox[3] - bbox[1]), 1)
        canvas_h = max(int(self.results_canvas.winfo_height()), 1)
        max_scroll = max(inner_h - canvas_h, 1)
        try:
            target_y = max(widget.winfo_rooty() - self.results_inner.winfo_rooty() - padding, 0)
        except Exception:
            target_y = max(widget.winfo_y() - padding, 0)
        self.results_canvas.yview_moveto(float(np.clip(target_y / max_scroll, 0.0, 1.0)))

    def _scroll_to_reg2d(self):
        self._scroll_results_to_widget(getattr(self, "reg2d_frame", None))

    def _open_image_popup(self, pil_img, title: str = "Image"):
        if pil_img is None:
            return
        top = self._tk.Toplevel(self.root)
        top.title(title)
        top.geometry("1400x920")

        ctrl = self._ttk.Frame(top)
        ctrl.pack(fill="x", padx=6, pady=4)
        self._ttk.Label(ctrl, text="缩放:").pack(side="left")
        state = {"mode": "fit"}

        canvas_frame = self._ttk.Frame(top)
        canvas_frame.pack(fill="both", expand=True)
        yscroll = self._ttk.Scrollbar(canvas_frame, orient="vertical")
        yscroll.pack(side="right", fill="y")
        xscroll = self._ttk.Scrollbar(canvas_frame, orient="horizontal")
        xscroll.pack(side="bottom", fill="x")
        canvas = self._tk.Canvas(
            canvas_frame, bg="#111111",
            xscrollcommand=xscroll.set, yscrollcommand=yscroll.set,
            highlightthickness=0,
        )
        canvas.pack(fill="both", expand=True)
        xscroll.config(command=canvas.xview)
        yscroll.config(command=canvas.yview)

        base_img = pil_img.copy()

        def _render_popup(_event=None):
            cw = max(canvas.winfo_width(), 400)
            ch = max(canvas.winfo_height(), 300)
            if state["mode"] == "fit":
                scale = min(cw / max(base_img.width, 1), ch / max(base_img.height, 1))
            else:
                scale = float(state["mode"])
            scale = max(scale, 0.05)
            w = max(int(base_img.width * scale), 1)
            h = max(int(base_img.height * scale), 1)
            resized = base_img.resize((w, h), self._PILImage.Resampling.LANCZOS)
            photo = self._ImageTk.PhotoImage(resized)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor="nw", image=photo)
            canvas._photo = photo
            canvas.config(scrollregion=(0, 0, w, h))

        def _set_zoom(mode):
            state["mode"] = mode
            _render_popup()

        for label, mode in (("适应窗口", "fit"), ("100%", 1.0), ("150%", 1.5), ("200%", 2.0)):
            self._ttk.Button(ctrl, text=label, command=lambda m=mode: _set_zoom(m)).pack(side="left", padx=(4, 0))

        canvas.bind("<Configure>", _render_popup)
        _render_popup()

    def _open_canvas_popup(self, canvas):
        pil_img = getattr(canvas, "_full_pil", None)
        title = getattr(canvas, "_popup_title", "Image")
        if pil_img is not None:
            self._open_image_popup(pil_img, title=title)

    def _set_viewer_enabled(self, enabled: bool):
        self.view3d_btn.config(state="normal" if enabled else "disabled")

    def _set_download_path(self, path: str | None, result_dir: str | None = None):
        self._download_path = path if path and os.path.exists(path) else None
        self._result_dir_path = result_dir if result_dir and os.path.isdir(result_dir) else None
        if self._download_path:
            self.download_var.set(f"下载包: {os.path.basename(self._download_path)}")
            self.download_btn.config(state="normal")
        else:
            self.download_var.set("下载包: 暂无")
            self.download_btn.config(state="disabled")
        if self._result_dir_path:
            self.result_dir_btn.config(state="normal")
        else:
            self.result_dir_btn.config(state="disabled")

        if self._download_path and self._result_dir_path:
            self.action_hint_var.set(
                f"可操作: 下载 {os.path.basename(self._download_path)} / 打开结果目录 / 3D 查看"
            )
        elif self._result_dir_path:
            self.action_hint_var.set("已生成结果目录，可打开目录查看输出")
        else:
            self.action_hint_var.set("运行完成后，这里会提供下载、目录和 3D 查看入口")

    def _open_path(self, path: str | None):
        if not path or not os.path.exists(path):
            self.run_status_var.set("文件不存在")
            return
        try:
            if sys.platform == "darwin":
                subprocess.Popen(["open", path])
            elif os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", path])
        except Exception as e:
            self.run_status_var.set(f"打开失败: {e}")

    def _open_download_path(self, _event=None):
        self._open_path(self._download_path)

    def _open_result_dir(self):
        self._open_path(self._result_dir_path)

    def _get_selected_idx(self) -> str | None:
        sel = self.listbox.curselection()
        return self.indices[sel[0]] if sel else None

    def _refresh_selection_summary(self):
        before = self.before_idx or "-"
        after = self.after_idx or "-"
        self.before_var.set(f"Before: {before}")
        self.after_var.set(f"After: {after}")
        self.selection_summary_var.set(
            f"当前配对: pick前 = {before}   |   pick后 = {after}"
        )

    def _refresh_before_preview(self):
        if self.before_idx is None:
            self.before_canvas.delete("all")
            self.before_canvas.create_text(10, 10, anchor="nw", text="未选择 Before", fill="gray")
            return
        self._show_on_canvas(
            os.path.join(self.data_dir, f"{self.before_idx}_origin_L.png"),
            self.before_canvas,
            f"Before: {self.before_idx}",
        )

    def _refresh_after_preview(self):
        if self.after_idx is None:
            self.after_canvas.delete("all")
            self.after_canvas.create_text(10, 10, anchor="nw", text="未选择 After", fill="gray")
            return
        self._show_on_canvas(
            os.path.join(self.data_dir, f"{self.after_idx}_origin_L.png"),
            self.after_canvas,
            f"After: {self.after_idx}",
        )

    def _show_on_canvas(self, img_path_or_array, canvas, label: str = ""):
        try:
            if isinstance(img_path_or_array, str):
                pil = self._PILImage.open(img_path_or_array)
            else:
                arr = np.asarray(img_path_or_array)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                pil = self._PILImage.fromarray(arr)
            full_pil = pil.copy()
            canvas.update_idletasks()
            cw = max(canvas.winfo_width(), 200)
            ch = max(canvas.winfo_height(), 120)
            ratio = min(cw / pil.width, ch / pil.height)
            pil = pil.resize((int(pil.width * ratio), int(pil.height * ratio)),
                             self._PILImage.Resampling.LANCZOS)
            photo = self._ImageTk.PhotoImage(pil)
            canvas.delete("all")
            canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
            if label:
                canvas.create_text(5, 5, anchor="nw", text=label, fill="#1b5e20",
                                   font=("", 10, "bold"))
            canvas._photo = photo
            canvas._full_pil = full_pil
            canvas._popup_title = label or "Image"
            canvas.bind("<Double-Button-1>", lambda _event, c=canvas: self._open_canvas_popup(c))
        except Exception as e:
            canvas.delete("all")
            canvas.create_text(10, 10, anchor="nw", text=f"Error: {e}", fill="red")

    def _on_list_select(self, _event=None):
        idx = self._get_selected_idx()
        if idx is None:
            return
        img_path = os.path.join(self.data_dir, f"{idx}_origin_L.png")
        self._show_on_canvas(img_path, self.preview_canvas, f"序号 {idx}")
        self.preview_info_var.set(
            f"当前高亮: {idx}   |   图像: {idx}_origin_L.png / {idx}_origin_R.png"
        )
        self.run_status_var.set(f"已高亮序号 {idx}")

    def _set_before(self):
        idx = self._get_selected_idx()
        if idx is None:
            self.run_status_var.set("请先在列表中选择一个序号")
            return
        self.before_idx = idx
        self._refresh_before_preview()
        self._refresh_selection_summary()
        self.run_status_var.set(f"已设置 pick前 序号: {idx}")

    def _set_after(self):
        idx = self._get_selected_idx()
        if idx is None:
            self.run_status_var.set("请先在列表中选择一个序号")
            return
        self.after_idx = idx
        self._refresh_after_preview()
        self._refresh_selection_summary()
        self.run_status_var.set(f"已设置 pick后 序号: {idx}")

    def _on_run(self):
        if self.before_idx is None or self.after_idx is None:
            self.run_status_var.set("请先选择 Before 和 After 序号")
            return
        if self.before_idx == self.after_idx:
            self.run_status_var.set("Before 和 After 不能是同一组数据")
            return

        self.run_status_var.set(f"运行中...")
        self._set_viewer_enabled(False)
        self._last_run_pcds = None
        self._last_run_info = ""
        self._log_to_text(
            f"开始运行  Before={self.before_idx}  After={self.after_idx}\n"
            f"数据目录: {self.data_dir}\n"
            f"内参文件: {self.intrinsic_path}\n"
            + "=" * 50
        )
        self._set_detection_text("等待检测结果...")
        self._set_shape_text("等待形状拟合结果...")
        self.before_shape_canvas.delete("all")
        self.before_shape_canvas.create_text(12, 12, anchor="nw", text="等待 Before shape fit...", fill="gray")
        self.after_shape_canvas.delete("all")
        self.after_shape_canvas.create_text(12, 12, anchor="nw", text="等待 After shape fit...", fill="gray")
        self.reg2d_canvas.delete("all")
        self.reg2d_canvas.create_text(12, 12, anchor="nw", text="等待二维叠加结果...", fill="gray")
        self._set_download_path(None, None)
        if hasattr(self, "results_canvas"):
            self.results_canvas.yview_moveto(0.0)
        self.root.update()

        before_l = os.path.join(self.data_dir, f"{self.before_idx}_origin_L.png")
        before_r = os.path.join(self.data_dir, f"{self.before_idx}_origin_R.png")
        after_l = os.path.join(self.data_dir, f"{self.after_idx}_origin_L.png")
        after_r = os.path.join(self.data_dir, f"{self.after_idx}_origin_R.png")

        try:
            ret = run_local(
                before_left_path=before_l, before_right_path=before_r,
                after_left_path=after_l, after_right_path=after_r,
                intrinsic_path=self.intrinsic_path,
                object_name=self.obj_name_var.get().strip(),
                sam3_url=self.sam3_url, s2m2_url=self.s2m2_url,
                voxel_size=self.voxel_size,
                use_coarse_fine=self.use_coarse_fine,
                parent=self.root,
                show_viewer=False,
                log_fn=lambda msg: self._log_to_text(msg, append=True),
            )

            result = ret["result"]
            self._last_run_pcds = ret["pcds"]
            self._last_run_info = "\n".join(ret["info_lines"])

            if ret.get("before_mask_vis") is not None:
                self._show_on_canvas(ret["before_mask_vis"], self.before_mask_canvas, "Before")
            if ret.get("after_mask_vis") is not None:
                self._show_on_canvas(ret["after_mask_vis"], self.after_mask_canvas, "After")
            if ret.get("reg2d_vis") is not None:
                self._show_on_canvas(ret["reg2d_vis"], self.reg2d_canvas, "2D Overlay")
            self._set_detection_text(ret.get("detection_summary") or "无检测诊断")
            if ret.get("before_shape_vis") is not None:
                self._show_on_canvas(ret["before_shape_vis"], self.before_shape_canvas, "Before shape fit")
            if ret.get("after_shape_vis") is not None:
                self._show_on_canvas(ret["after_shape_vis"], self.after_shape_canvas, "After shape fit")
            self._set_shape_text(ret.get("shape_summary") or "无形状拟合结果")
            self._set_download_path(ret.get("bundle_path"), ret.get("save_dir"))
            self._set_viewer_enabled(ret.get("pcds") is not None)

            if ret.get("success") and result is not None:
                q = result.debug.get("quality", {}).get("quality", "?")
                self.run_status_var.set(
                    f"完成 | {q.upper()} | {result.elapsed_s:.2f}s | 保存: {ret['save_dir']}"
                )
                if ret.get("reg2d_vis") is not None:
                    self.root.after(80, self._scroll_to_reg2d)
            else:
                self.run_status_var.set(f"失败: {ret.get('error') or '未知错误'}")
        except Exception as e:
            self._set_viewer_enabled(False)
            self._set_download_path(None, None)
            self._set_shape_text("运行异常，未生成形状拟合结果")
            self.run_status_var.set(f"失败: {e}")
            self._log_to_text(f"\n{'='*50}\n失败: {e}\n{traceback.format_exc()}", append=True)

    def _open_3d_viewer(self):
        if self._last_run_pcds is None:
            return
        pcds = self._last_run_pcds
        viewer = _LocalResultViewer(
            pcds["before"], pcds["after"], pcds["aligned"],
            after_scene_pcd=pcds.get("after_scene"),
            info_text=getattr(self, "_last_run_info", ""),
            parent=self.root,
        )
        viewer.run()

    def run(self):
        self.root.mainloop()


# ============================================================================
# Local launcher GUI (tkinter file picker)
# ============================================================================

class _LocalLauncher:
    """Tkinter window with file pickers for all inputs, then runs pipeline + viewer."""

    def __init__(self):
        import tkinter as _tk
        from tkinter import filedialog as _fd, ttk as _ttk

        self.root = _tk.Tk()
        self.root.title("In-Hand Pose Re-Estimation — Local Launcher")
        self.root.geometry("680x520")
        self.root.resizable(True, True)

        self._tk = _tk
        self._fd = _fd

        main = _ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        _ttk.Label(main, text="Post-Grasp In-Hand Pose Re-Estimation", font=("", 14, "bold")).pack(pady=(0, 8))

        # File path variables
        self.before_left_var = _tk.StringVar()
        self.before_right_var = _tk.StringVar()
        self.after_left_var = _tk.StringVar()
        self.after_right_var = _tk.StringVar()
        self.intrinsic_var = _tk.StringVar()
        self.object_name_var = _tk.StringVar(value="")
        self.sam3_url_var = _tk.StringVar(value=DEFAULT_SAM3_URL)
        self.s2m2_url_var = _tk.StringVar(value=DEFAULT_S2M2_URL)
        self.voxel_var = _tk.StringVar(value="0.003")
        self.coarse_fine_var = _tk.BooleanVar(value=True)
        self.status_var = _tk.StringVar(value="选择文件后点击 [运行]")

        file_types = [("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        txt_types = [("Text files", "*.txt"), ("All files", "*.*")]

        def _add_file_row(parent, label, var, ftypes):
            row = _ttk.Frame(parent)
            row.pack(fill="x", pady=2)
            _ttk.Label(row, text=label, width=16, anchor="w").pack(side="left")
            _ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True, padx=(4, 4))
            _ttk.Button(row, text="浏览...", width=6,
                        command=lambda: var.set(self._fd.askopenfilename(filetypes=ftypes) or var.get())
                        ).pack(side="right")

        _ttk.Label(main, text="抓取前 双目图像", font=("", 10, "bold"), anchor="w").pack(fill="x", pady=(8, 0))
        _add_file_row(main, "左图:", self.before_left_var, file_types)
        _add_file_row(main, "右图:", self.before_right_var, file_types)

        _ttk.Label(main, text="抓取后 双目图像", font=("", 10, "bold"), anchor="w").pack(fill="x", pady=(8, 0))
        _add_file_row(main, "左图:", self.after_left_var, file_types)
        _add_file_row(main, "右图:", self.after_right_var, file_types)

        _ttk.Label(main, text="相机内参", font=("", 10, "bold"), anchor="w").pack(fill="x", pady=(8, 0))
        _add_file_row(main, "内参文件:", self.intrinsic_var, txt_types)

        _ttk.Label(main, text="参数", font=("", 10, "bold"), anchor="w").pack(fill="x", pady=(8, 0))
        params = _ttk.Frame(main)
        params.pack(fill="x", pady=2)
        _ttk.Label(params, text="物体名称:").pack(side="left")
        _ttk.Entry(params, textvariable=self.object_name_var, width=16).pack(side="left", padx=(4, 12))
        _ttk.Label(params, text="Voxel:").pack(side="left")
        _ttk.Entry(params, textvariable=self.voxel_var, width=8).pack(side="left", padx=(4, 12))
        _ttk.Checkbutton(params, text="Coarse-to-Fine", variable=self.coarse_fine_var).pack(side="left")

        urls = _ttk.Frame(main)
        urls.pack(fill="x", pady=2)
        _ttk.Label(urls, text="S2M2:").pack(side="left")
        _ttk.Entry(urls, textvariable=self.s2m2_url_var, width=32).pack(side="left", padx=(4, 8))
        _ttk.Label(urls, text="SAM3:").pack(side="left")
        _ttk.Entry(urls, textvariable=self.sam3_url_var, width=32).pack(side="left", padx=(4, 0))

        btn_row = _ttk.Frame(main)
        btn_row.pack(fill="x", pady=(12, 4))
        _ttk.Button(btn_row, text="运行", command=self._on_run).pack(side="left", padx=(0, 8))
        _ttk.Label(btn_row, textvariable=self.status_var, foreground="gray").pack(side="left", fill="x", expand=True)

    def _on_run(self):
        paths = {
            "before_left": self.before_left_var.get().strip(),
            "before_right": self.before_right_var.get().strip(),
            "after_left": self.after_left_var.get().strip(),
            "after_right": self.after_right_var.get().strip(),
            "intrinsic": self.intrinsic_var.get().strip(),
        }
        missing = [k for k, v in paths.items() if not v]
        if missing:
            self.status_var.set(f"缺少: {', '.join(missing)}")
            return
        for k, v in paths.items():
            if not os.path.isfile(v):
                self.status_var.set(f"文件不存在: {v}")
                return

        self.status_var.set("运行中...")
        self.root.update()

        try:
            ret = run_local(
                before_left_path=paths["before_left"],
                before_right_path=paths["before_right"],
                after_left_path=paths["after_left"],
                after_right_path=paths["after_right"],
                intrinsic_path=paths["intrinsic"],
                object_name=self.object_name_var.get().strip(),
                sam3_url=self.sam3_url_var.get().strip(),
                s2m2_url=self.s2m2_url_var.get().strip(),
                voxel_size=float(self.voxel_var.get()),
                use_coarse_fine=self.coarse_fine_var.get(),
            )
            if ret.get("success"):
                self.status_var.set("完成")
            else:
                self.status_var.set(f"失败: {ret.get('error') or '未知错误'}")
        except Exception as e:
            self.status_var.set(f"失败: {e}")
            traceback.print_exc()

    def run(self):
        self.root.mainloop()


# ============================================================================
# Entry
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-Grasp In-Hand Pose Re-Estimation")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7880)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--local", action="store_true",
                        help="本地模式: 浏览 stereonet_result 序号列表并运行")
    parser.add_argument("--local-files", action="store_true",
                        help="本地文件模式: 手动选择左右图和内参")
    parser.add_argument("--browse", action="store_true",
                        help="兼容旧参数: 等同于 --local")
    parser.add_argument("--data-dir", default="stereonet_result",
                        help="stereonet_result 数据目录路径 (默认: stereonet_result)")
    parser.add_argument("--object-name", default="", help="目标物体名称 (用于检测)")

    args = parser.parse_args()

    if args.local or args.browse:
        browser = _StereoNetBrowser(
            data_dir=args.data_dir,
            object_name=args.object_name,
        )
        browser.run()
    elif args.local_files:
        launcher = _LocalLauncher()
        launcher.run()
    else:
        app = build_app()
        app.launch(
            server_name=args.host, server_port=args.port,
            share=args.share, css=CUSTOM_CSS,
        )
