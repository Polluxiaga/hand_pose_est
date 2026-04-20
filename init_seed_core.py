from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SeedCandidate:
    tag: str
    T_init: np.ndarray
    score_hint: float = 0.0
    search_mode: str = "full"
    metadata: dict[str, Any] = field(default_factory=dict)


def reconstruct_init_transform(result: Any) -> np.ndarray | None:
    if result is None:
        return None
    T = getattr(result, "T_before_cam_to_after_cam", None)
    if T is not None:
        return np.asarray(T, dtype=np.float64).reshape(4, 4)
    debug = getattr(result, "debug", {}) or {}
    rot = debug.get("prior_rotation")
    trans = debug.get("prior_translation")
    if rot is None or trans is None:
        return None
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(trans, dtype=np.float64).reshape(3)
    return T


def registration_working_target_points(seed_policy: str) -> int:
    policy = str(seed_policy or "")
    if policy in {"tapir_prior_direct", "sam3d_prior_direct"}:
        return 0
    if policy in {"tapir_prior_compare", "sam3d_prior_compare", "sam3d_single_seed"}:
        return 5000
    if policy == "fallback_search":
        return 6000
    return 0


def prior_keep_debug(candidate: SeedCandidate) -> dict[str, Any]:
    metadata = candidate.metadata or {}
    prior_rmse = float(metadata.get("prior_dense_rmse_eval", math.inf))
    prior_fitness = float(metadata.get("prior_dense_fitness_eval", 0.0))
    prior_proj_iou = float(metadata.get("prior_proj_iou", 0.0))
    keep_tag = str(candidate.tag or "")
    return {
        "search_mode": str(candidate.search_mode or keep_tag),
        "coarse_score": float("nan"),
        "coarse_ms": 0,
        "coarse_skipped": True,
        "fine_score": float("nan"),
        "fine_ms": 0,
        "gicp_fitness": prior_fitness,
        "gicp_inlier_rmse": prior_rmse,
        "gicp_visibility_score": 0.0,
        "gicp_projection_iou": prior_proj_iou,
        "gicp_ms": 0,
        "colored_icp_disabled": True,
        "colored_icp_rejected": True,
        "colored_icp_reject_reasons": [keep_tag or "prior_keep"],
        "colored_icp_effective_source": keep_tag or "prior_keep",
        "colored_icp_fitness": prior_fitness,
        "colored_icp_inlier_rmse": prior_rmse,
        "colored_icp_ms": 0,
        "total_reg_ms": 0,
    }


def _compute_axes_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return np.eye(3, dtype=np.float64), pts.mean(axis=0) if len(pts) else np.zeros(3, dtype=np.float64)
    center = pts.mean(axis=0)
    pts0 = pts - center
    _, _, vh = np.linalg.svd(pts0, full_matrices=False)
    axes = vh.T
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return axes.astype(np.float64), center.astype(np.float64)


def _compute_current_pca_obb(sample: dict[str, Any]) -> dict[str, Any]:
    points = np.asarray(sample.get("object_xyz", np.zeros((0, 3), dtype=np.float32)), dtype=np.float64)
    axes, center = _compute_axes_from_points(points)
    return {
        "axes": axes,
        "center_m": center,
    }


def _rotation_delta_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    R_rel = np.asarray(R_a, dtype=np.float64).reshape(3, 3) @ np.asarray(R_b, dtype=np.float64).reshape(3, 3).T
    trace = float(np.trace(R_rel))
    cos_theta = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def _transform_from_axes(ref_axes: np.ndarray, ref_center: np.ndarray, cur_axes: np.ndarray, cur_center: np.ndarray) -> np.ndarray:
    R = np.asarray(cur_axes, dtype=np.float64).reshape(3, 3) @ np.asarray(ref_axes, dtype=np.float64).reshape(3, 3).T
    t = np.asarray(cur_center, dtype=np.float64).reshape(3) - (R @ np.asarray(ref_center, dtype=np.float64).reshape(3))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _build_axis_permutation_candidates(ref_obb: dict[str, Any], cur_obb: dict[str, Any]) -> list[SeedCandidate]:
    ref_axes = np.asarray(ref_obb.get("axes", np.eye(3)), dtype=np.float64).reshape(3, 3)
    ref_center = np.asarray(ref_obb.get("center_m", np.zeros(3)), dtype=np.float64).reshape(3)
    cur_axes = np.asarray(cur_obb.get("axes", np.eye(3)), dtype=np.float64).reshape(3, 3)
    cur_center = np.asarray(cur_obb.get("center_m", np.zeros(3)), dtype=np.float64).reshape(3)
    candidates: list[SeedCandidate] = []
    seen: list[np.ndarray] = []
    basis = np.eye(3, dtype=np.float64)
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        P = basis[:, perm]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                S = np.diag([sx, sy, 1.0])
                cand_axes = cur_axes @ P @ S
                if np.linalg.det(cand_axes) < 0.0:
                    cand_axes[:, -1] *= -1.0
                T = _transform_from_axes(ref_axes, ref_center, cand_axes, cur_center)
                R = T[:3, :3]
                duplicate = any(np.allclose(R, prev, atol=1e-6) for prev in seen)
                if duplicate:
                    continue
                seen.append(R.copy())
                candidates.append(
                    SeedCandidate(
                        tag="pca_perm",
                        T_init=T,
                        score_hint=0.02,
                        search_mode="fallback_light",
                        metadata={
                            "perm": list(perm),
                            "sign_x": sx,
                            "sign_y": sy,
                        },
                    )
                )
    return candidates


def _seed_rank_score(candidate: SeedCandidate, reference_T: np.ndarray | None) -> float:
    T = np.asarray(candidate.T_init, dtype=np.float64).reshape(4, 4)
    if reference_T is None:
        rot_score = _rotation_delta_deg(T[:3, :3], np.eye(3, dtype=np.float64))
        trans_score = float(np.linalg.norm(T[:3, 3]))
    else:
        ref_T = np.asarray(reference_T, dtype=np.float64).reshape(4, 4)
        rot_score = _rotation_delta_deg(T[:3, :3], ref_T[:3, :3])
        trans_score = float(np.linalg.norm(T[:3, 3] - ref_T[:3, 3]))
    return float(rot_score + trans_score * 1000.0)


def _candidate_signature(candidate: SeedCandidate) -> tuple[float, ...]:
    T = np.asarray(candidate.T_init, dtype=np.float64).reshape(4, 4)
    mode_signature = float(sum(ord(ch) for ch in str(candidate.search_mode or "")))
    return tuple(np.round(np.concatenate([[mode_signature], T[:3, :3].reshape(-1), T[:3, 3]]), 4).tolist())


def _append_unique_seed(candidates: list[SeedCandidate], candidate: SeedCandidate | None, seen: set[tuple[float, ...]]) -> bool:
    if candidate is None:
        return False
    signature = _candidate_signature(candidate)
    if signature in seen:
        return False
    seen.add(signature)
    candidates.append(candidate)
    return True


def _take_ranked_candidates(
    base_candidates: list[SeedCandidate],
    limit: int,
    reference_T: np.ndarray | None,
) -> list[SeedCandidate]:
    if limit <= 0:
        return []
    ranked = sorted(base_candidates, key=lambda candidate: (_seed_rank_score(candidate, reference_T), candidate.tag))
    return ranked[:limit]


def _build_fallback_seed_candidates(
    ref_obb: dict[str, Any],
    cur_obb: dict[str, Any],
    reference_T: np.ndarray | None,
    candidate_limit: int = 16,
) -> tuple[list[SeedCandidate], dict[str, int]]:
    raw_pca_candidates = _build_axis_permutation_candidates(ref_obb, cur_obb)
    selected_pca_candidates = _take_ranked_candidates(
        raw_pca_candidates,
        limit=int(candidate_limit),
        reference_T=reference_T,
    )
    return selected_pca_candidates, {
        "pca_candidate_count": int(len(raw_pca_candidates)),
        "pca_candidates_used": int(len(selected_pca_candidates)),
    }


def _build_tapir_prior_keep_candidate(tapir_T: np.ndarray, tapir_debug: dict[str, Any]) -> SeedCandidate:
    return SeedCandidate(
        tag="tapir_prior_keep",
        T_init=np.asarray(tapir_T, dtype=np.float64).reshape(4, 4),
        score_hint=-0.24,
        search_mode="tapir_prior_keep",
        metadata={
            "prior_dense_rmse_eval": float((tapir_debug or {}).get("prior_dense_rmse_eval", math.inf)),
            "prior_dense_fitness_eval": float((tapir_debug or {}).get("prior_dense_fitness_eval", 0.0)),
            "prior_proj_iou": float((tapir_debug or {}).get("prior_proj_iou", 0.0)),
            "prior_proj_precision": float((tapir_debug or {}).get("prior_proj_precision", 0.0)),
        },
    )


def _build_sam3d_prior_keep_candidate(sam3d_T: np.ndarray, sam3d_debug: dict[str, Any]) -> SeedCandidate:
    return SeedCandidate(
        tag="sam3d_prior_keep",
        T_init=np.asarray(sam3d_T, dtype=np.float64).reshape(4, 4),
        score_hint=-0.20,
        search_mode="sam3d_prior_keep",
        metadata={
            "prior_dense_rmse_eval": float((sam3d_debug or {}).get("prior_dense_rmse_eval", math.inf)),
            "prior_dense_fitness_eval": float((sam3d_debug or {}).get("prior_dense_fitness_eval", 0.0)),
            "prior_proj_iou": float((sam3d_debug or {}).get("prior_proj_iou", 0.0)),
            "prior_proj_precision": float((sam3d_debug or {}).get("prior_proj_precision", 0.0)),
        },
    )


def _tapir_compare_refine_search_mode(tapir_debug: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    prior_confidence = float((tapir_debug or {}).get("prior_confidence", 0.0) or 0.0)
    proj_iou = float((tapir_debug or {}).get("prior_proj_iou", 0.0) or 0.0)
    proj_precision = float((tapir_debug or {}).get("prior_proj_precision", 0.0) or 0.0)
    rel_dense_rmse = float((tapir_debug or {}).get("prior_dense_relative_rmse_eval", math.inf) or math.inf)

    verify_ready = bool(
        prior_confidence >= 0.56
        and proj_iou >= 0.45
        and proj_precision >= 0.60
        and np.isfinite(rel_dense_rmse)
        and rel_dense_rmse <= 0.065
    )
    if verify_ready:
        return "tapir_micro_verify_strict", {
            "compare_refine_mode": "micro_verify_strict",
            "compare_refine_reason": "high_medium_confidence_prior_direct_gicp_verify",
        }
    return "tapir_micro_verify_relaxed", {
        "compare_refine_mode": "micro_verify_relaxed",
        "compare_refine_reason": "default_medium_band_direct_gicp_verify",
    }


def _sam3d_compare_refine_search_mode(sam3d_debug: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    prior_confidence = float((sam3d_debug or {}).get("prior_confidence", 0.0) or 0.0)
    proj_iou = float((sam3d_debug or {}).get("prior_proj_iou", 0.0) or 0.0)
    proj_precision = float((sam3d_debug or {}).get("prior_proj_precision", 0.0) or 0.0)
    rel_dense_rmse = float((sam3d_debug or {}).get("prior_dense_relative_rmse_eval", math.inf) or math.inf)

    verify_ready = bool(
        prior_confidence >= 0.56
        and proj_iou >= 0.45
        and proj_precision >= 0.60
        and np.isfinite(rel_dense_rmse)
        and rel_dense_rmse <= 0.065
    )
    if verify_ready:
        return "sam3d_micro_verify_strict", {
            "compare_refine_mode": "micro_verify_strict",
            "compare_refine_reason": "sam3d_high_medium_confidence_prior_direct_gicp_verify",
        }
    return "sam3d_micro_verify_relaxed", {
        "compare_refine_mode": "micro_verify_relaxed",
        "compare_refine_reason": "sam3d_default_medium_band_direct_gicp_verify",
    }


def _tapir_compare_entry_decision(tapir_debug: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    prior_confidence = float((tapir_debug or {}).get("prior_confidence", 0.0) or 0.0)
    proj_iou = float((tapir_debug or {}).get("prior_proj_iou", 0.0) or 0.0)
    proj_precision = float((tapir_debug or {}).get("prior_proj_precision", 0.0) or 0.0)
    rel_dense_rmse = float((tapir_debug or {}).get("prior_dense_relative_rmse_eval", math.inf) or math.inf)
    inliers = int((tapir_debug or {}).get("num_inliers", (tapir_debug or {}).get("inlier_count", 0)) or 0)

    compare_verify_confidence_thresh = 0.56
    low_inlier_thresh = 4
    low_proj_iou_thresh = 0.36
    low_proj_precision_thresh = 0.52
    weak_projection_support = bool(
        proj_iou < low_proj_iou_thresh
        and proj_precision < low_proj_precision_thresh
    )
    weak_medium_prior = bool(
        prior_confidence < compare_verify_confidence_thresh
        and inliers <= low_inlier_thresh
        and weak_projection_support
    )
    return (not weak_medium_prior), {
        "compare_entry_gate_ok": bool(not weak_medium_prior),
        "compare_entry_reason": (
            "weak_medium_prior_low_projection_support" if weak_medium_prior else "tapir_compare_allowed"
        ),
        "compare_entry_verify_confidence_thresh": float(compare_verify_confidence_thresh),
        "compare_entry_low_inlier_thresh": int(low_inlier_thresh),
        "compare_entry_low_proj_iou_thresh": float(low_proj_iou_thresh),
        "compare_entry_low_proj_precision_thresh": float(low_proj_precision_thresh),
        "compare_entry_prior_confidence": float(prior_confidence),
        "compare_entry_proj_iou": float(proj_iou),
        "compare_entry_proj_precision": float(proj_precision),
        "compare_entry_rel_dense_rmse": float(rel_dense_rmse),
        "compare_entry_inliers": int(inliers),
    }


def _sam3d_compare_entry_decision(sam3d_debug: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    prior_confidence = float((sam3d_debug or {}).get("prior_confidence", 0.0) or 0.0)
    proj_iou = float((sam3d_debug or {}).get("prior_proj_iou", 0.0) or 0.0)
    proj_precision = float((sam3d_debug or {}).get("prior_proj_precision", 0.0) or 0.0)
    rel_dense_rmse = float((sam3d_debug or {}).get("prior_dense_relative_rmse_eval", math.inf) or math.inf)

    compare_verify_confidence_thresh = 0.56
    low_proj_iou_thresh = 0.36
    low_proj_precision_thresh = 0.52
    high_rel_dense_rmse_thresh = 0.085
    weak_medium_prior = bool(
        prior_confidence < compare_verify_confidence_thresh
        and proj_iou < low_proj_iou_thresh
        and proj_precision < low_proj_precision_thresh
        and (not np.isfinite(rel_dense_rmse) or rel_dense_rmse > high_rel_dense_rmse_thresh)
    )
    return (not weak_medium_prior), {
        "compare_entry_gate_ok": bool(not weak_medium_prior),
        "compare_entry_reason": (
            "weak_medium_prior_low_projection_support" if weak_medium_prior else "sam3d_compare_allowed"
        ),
        "compare_entry_verify_confidence_thresh": float(compare_verify_confidence_thresh),
        "compare_entry_low_proj_iou_thresh": float(low_proj_iou_thresh),
        "compare_entry_low_proj_precision_thresh": float(low_proj_precision_thresh),
        "compare_entry_high_rel_dense_rmse_thresh": float(high_rel_dense_rmse_thresh),
        "compare_entry_prior_confidence": float(prior_confidence),
        "compare_entry_proj_iou": float(proj_iou),
        "compare_entry_proj_precision": float(proj_precision),
        "compare_entry_rel_dense_rmse": float(rel_dense_rmse),
    }


def build_tapir_seed_candidates(
    before: dict[str, Any],
    after: dict[str, Any],
    tapir_raw: Any,
    tapir_debug: dict[str, Any],
) -> tuple[list[SeedCandidate], dict[str, Any]]:
    candidates: list[SeedCandidate] = []
    seed_debug: dict[str, Any] = {}
    seen: set[tuple[float, ...]] = set()
    tapir_T = reconstruct_init_transform(tapir_raw)
    tapir_gate_ok = bool((tapir_debug or {}).get("prior_gate_decision", False))
    tapir_strong_ok = bool((tapir_debug or {}).get("prior_gate_strong_3d_ok", False))
    tapir_borderline_ok = bool((tapir_debug or {}).get("prior_gate_borderline_override", False))
    tapir_compare_search_mode, tapir_compare_mode_debug = _tapir_compare_refine_search_mode(tapir_debug or {})
    tapir_compare_entry_ok, tapir_compare_entry_debug = _tapir_compare_entry_decision(tapir_debug or {})
    tapir_seed = None
    if tapir_T is not None:
        tapir_seed = SeedCandidate(
            tag="tapir_prior_local_refine",
            T_init=tapir_T,
            score_hint=(-0.20 if tapir_strong_ok else (-0.08 if tapir_gate_ok else 0.18)),
            search_mode=tapir_compare_search_mode,
            metadata=dict(tapir_compare_mode_debug),
        )

    ref_obb = before.get("object_obb") or _compute_current_pca_obb(before)
    cur_obb = after.get("object_obb") or _compute_current_pca_obb(after)
    fallback_candidates, fallback_seed_stats = _build_fallback_seed_candidates(ref_obb, cur_obb, tapir_T)
    tapir_proj_iou = float((tapir_debug or {}).get("prior_proj_iou", 0.0))
    tapir_inliers = int((tapir_debug or {}).get("num_inliers", (tapir_debug or {}).get("inlier_count", 0)) or 0)
    prior_confidence = float((tapir_debug or {}).get("prior_confidence", 0.0) or 0.0)
    prior_confidence_accept = float((tapir_debug or {}).get("prior_confidence_accept_thresh", 0.0) or 0.0)
    prior_confidence_strong = float((tapir_debug or {}).get("prior_confidence_strong_thresh", 1.0) or 1.0)
    confidence_tier = "fallback"
    compare_entry_routed_to_fallback = bool(
        tapir_T is not None
        and tapir_gate_ok
        and tapir_seed is not None
        and not tapir_strong_ok
        and not tapir_compare_entry_ok
    )

    seed_policy = "fallback_search"
    if tapir_T is not None and tapir_strong_ok:
        seed_policy = "tapir_prior_direct"
        confidence_tier = "strong"
        _append_unique_seed(candidates, _build_tapir_prior_keep_candidate(tapir_T, tapir_debug), seen)
    elif tapir_T is not None and tapir_gate_ok and tapir_seed is not None and tapir_compare_entry_ok:
        seed_policy = "tapir_prior_compare"
        confidence_tier = "borderline" if tapir_borderline_ok and prior_confidence < prior_confidence_accept else "medium"
        _append_unique_seed(candidates, _build_tapir_prior_keep_candidate(tapir_T, tapir_debug), seen)
        _append_unique_seed(candidates, tapir_seed, seen)
    else:
        if compare_entry_routed_to_fallback:
            confidence_tier = "medium_routed_fallback"
        for candidate in fallback_candidates:
            _append_unique_seed(candidates, candidate, seen)
    seed_debug.update(
        {
            "seed_policy": seed_policy,
            "candidate_ranking_topk": [],
            **fallback_seed_stats,
            "tapir_gate_ok": tapir_gate_ok,
            "tapir_strong_ok": tapir_strong_ok,
            "tapir_borderline_ok": tapir_borderline_ok,
            "tapir_proj_iou": tapir_proj_iou,
            "tapir_inliers": tapir_inliers,
            **tapir_compare_mode_debug,
            **tapir_compare_entry_debug,
            "compare_entry_routed_to_fallback": bool(compare_entry_routed_to_fallback),
            "tapir_prior_confidence": float(prior_confidence),
            "tapir_prior_confidence_accept_thresh": float(prior_confidence_accept),
            "tapir_prior_confidence_strong_thresh": float(prior_confidence_strong),
            "tapir_prior_confidence_tier": confidence_tier,
        }
    )
    return candidates, seed_debug


def build_sam3d_seed_candidates(
    before: dict[str, Any],
    after: dict[str, Any],
    sam3d_raw: Any,
    sam3d_debug: dict[str, Any],
) -> tuple[list[SeedCandidate], dict[str, Any]]:
    candidates: list[SeedCandidate] = []
    seed_debug: dict[str, Any] = {}
    seen: set[tuple[float, ...]] = set()
    sam3d_T = reconstruct_init_transform(sam3d_raw)
    sam3d_gate_ok = bool((sam3d_debug or {}).get("prior_gate_decision", False))
    sam3d_strong_ok = bool((sam3d_debug or {}).get("prior_gate_strong_3d_ok", False))
    sam3d_borderline_ok = bool((sam3d_debug or {}).get("prior_gate_borderline_override", False))
    sam3d_compare_search_mode, sam3d_compare_mode_debug = _sam3d_compare_refine_search_mode(sam3d_debug or {})
    sam3d_compare_entry_ok, sam3d_compare_entry_debug = _sam3d_compare_entry_decision(sam3d_debug or {})
    sam3d_seed = None
    if sam3d_T is not None:
        sam3d_seed = SeedCandidate(
            tag="sam3d_prior_local_refine",
            T_init=sam3d_T,
            score_hint=(-0.18 if sam3d_strong_ok else (-0.06 if sam3d_gate_ok else 0.18)),
            search_mode=sam3d_compare_search_mode,
            metadata=dict(sam3d_compare_mode_debug),
        )

    ref_obb = before.get("object_obb") or _compute_current_pca_obb(before)
    cur_obb = after.get("object_obb") or _compute_current_pca_obb(after)
    fallback_candidates, fallback_seed_stats = _build_fallback_seed_candidates(ref_obb, cur_obb, sam3d_T)
    prior_confidence = float((sam3d_debug or {}).get("prior_confidence", 0.0) or 0.0)
    prior_confidence_accept = float((sam3d_debug or {}).get("prior_confidence_accept_thresh", 0.0) or 0.0)
    prior_confidence_strong = float((sam3d_debug or {}).get("prior_confidence_strong_thresh", 1.0) or 1.0)
    confidence_tier = "fallback"
    compare_entry_routed_to_fallback = bool(
        sam3d_T is not None
        and sam3d_gate_ok
        and sam3d_seed is not None
        and not sam3d_strong_ok
        and not sam3d_compare_entry_ok
    )

    seed_policy = "fallback_search"
    if sam3d_T is not None and sam3d_strong_ok:
        seed_policy = "sam3d_prior_direct"
        confidence_tier = "strong"
        _append_unique_seed(candidates, _build_sam3d_prior_keep_candidate(sam3d_T, sam3d_debug), seen)
    elif sam3d_T is not None and sam3d_gate_ok and sam3d_seed is not None and sam3d_compare_entry_ok:
        seed_policy = "sam3d_prior_compare"
        confidence_tier = "borderline" if sam3d_borderline_ok and prior_confidence < prior_confidence_accept else "medium"
        _append_unique_seed(candidates, _build_sam3d_prior_keep_candidate(sam3d_T, sam3d_debug), seen)
        _append_unique_seed(candidates, sam3d_seed, seen)
    else:
        if compare_entry_routed_to_fallback:
            confidence_tier = "medium_routed_fallback"
        for candidate in fallback_candidates:
            _append_unique_seed(candidates, candidate, seen)
    seed_debug.update(
        {
            "seed_policy": seed_policy,
            "candidate_ranking_topk": [],
            **fallback_seed_stats,
            "sam3d_gate_ok": sam3d_gate_ok,
            "sam3d_strong_ok": sam3d_strong_ok,
            "sam3d_borderline_ok": sam3d_borderline_ok,
            "sam3d_proj_iou": float((sam3d_debug or {}).get("prior_proj_iou", 0.0)),
            "sam3d_proj_precision": float((sam3d_debug or {}).get("prior_proj_precision", 0.0)),
            **sam3d_compare_mode_debug,
            **sam3d_compare_entry_debug,
            "compare_entry_routed_to_fallback": bool(compare_entry_routed_to_fallback),
            "sam3d_prior_confidence": float(prior_confidence),
            "sam3d_prior_confidence_accept_thresh": float(prior_confidence_accept),
            "sam3d_prior_confidence_strong_thresh": float(prior_confidence_strong),
            "sam3d_prior_confidence_tier": confidence_tier,
        }
    )
    return candidates, seed_debug


__all__ = [
    "SeedCandidate",
    "build_sam3d_seed_candidates",
    "build_tapir_seed_candidates",
    "prior_keep_debug",
    "reconstruct_init_transform",
    "registration_working_target_points",
]
