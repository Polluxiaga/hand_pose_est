#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError(f"Failed to write image: {path}")


def _draw_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    output = np.asarray(image_rgb, dtype=np.uint8).copy()
    mask_bool = np.asarray(mask) > 0
    if mask_bool.ndim == 3:
        mask_bool = mask_bool[:, :, 0]
    color_arr = np.asarray(color, dtype=np.float32)
    output[mask_bool] = np.clip(0.6 * output[mask_bool].astype(np.float32) + 0.4 * color_arr, 0, 255).astype(np.uint8)
    edge = mask_bool.astype(np.uint8) - cv2.erode(mask_bool.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1)
    output[edge > 0] = color
    return output


def _resize_rgb_to_height(image_rgb: np.ndarray, target_height: int) -> np.ndarray:
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image_rgb.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("Image must be non-empty.")
    if height == target_height:
        return image_rgb.copy()
    scale = float(target_height) / float(height)
    target_width = max(1, int(round(width * scale)))
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR)


def _prepend_reference_panel_rgb(reference_rgb: np.ndarray, overlay_rgb: np.ndarray, gap_px: int = 8) -> np.ndarray:
    target_height = max(int(reference_rgb.shape[0]), int(overlay_rgb.shape[0]))
    reference_panel = _resize_rgb_to_height(reference_rgb, target_height)
    overlay_panel = _resize_rgb_to_height(overlay_rgb, target_height)
    canvas = np.full((target_height, int(reference_panel.shape[1]) + gap_px + int(overlay_panel.shape[1]), 3), 255, dtype=np.uint8)
    canvas[:, : reference_panel.shape[1]] = reference_panel
    start = int(reference_panel.shape[1]) + gap_px
    canvas[:, start : start + overlay_panel.shape[1]] = overlay_panel
    return canvas


def _make_overlay_bundle(after_rgb: np.ndarray, after_mask: np.ndarray, right_panel_rgb: np.ndarray) -> np.ndarray:
    left_panel = _draw_mask_overlay(after_rgb, after_mask, color=(255, 166, 77))
    cv2.putText(left_panel, "After target mask", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 230, 210), 2, cv2.LINE_AA)
    right_panel = np.asarray(right_panel_rgb, dtype=np.uint8).copy()
    divider = np.full((left_panel.shape[0], 10, 3), 18, dtype=np.uint8)
    return np.concatenate([left_panel, divider, right_panel], axis=1)


def _mask_from_pixels(image_shape: tuple[int, int], pixels_xy: np.ndarray) -> np.ndarray:
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    pixels = np.asarray(pixels_xy, dtype=np.int64).reshape(-1, 2)
    if len(pixels) == 0:
        return mask
    xs = pixels[:, 0]
    ys = pixels[:, 1]
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    mask[ys[valid], xs[valid]] = 255
    return mask


def _iter_case_dirs(root: Path) -> list[Path]:
    if (root / "result.json").is_file():
        return [root]
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _restore_case(case_dir: Path, replay_root: Path) -> int:
    result_path = case_dir / "result.json"
    reference_path = case_dir / "memory_reference_left.png"
    if not result_path.is_file() or not reference_path.is_file():
        return 0

    replay_id = str((_read_json(result_path).get("case") or {}).get("replay_id") or "")
    if not replay_id:
        return 0
    replay_dir = replay_root / replay_id
    replay_image_path = replay_dir / "image_left.png"
    replay_pixels_path = replay_dir / "current_object_selected_pixels_xy.npy"
    if not replay_image_path.is_file() or not replay_pixels_path.is_file():
        return 0

    reference_rgb = _read_rgb(reference_path)
    after_rgb = _read_rgb(replay_image_path)
    after_mask = _mask_from_pixels(after_rgb.shape[:2], np.load(replay_pixels_path))

    updated = 0
    target_paths = [case_dir / "left_replay_overlay.png"]
    target_paths.extend(sorted(case_dir.glob("left_replay_object_tapir_prior_overlay_*.png")))
    for target_path in target_paths:
        if not target_path.is_file():
            continue
        current_rgb = _read_rgb(target_path)
        if current_rgb.shape[1] > after_rgb.shape[1] * 2:
            continue
        overlay_bundle = _make_overlay_bundle(after_rgb, after_mask, current_rgb)
        restored_rgb = _prepend_reference_panel_rgb(reference_rgb, overlay_bundle)
        _write_rgb(target_path, restored_rgb)
        print(f"restored {target_path}")
        updated += 1
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore left_replay PNGs to three-panel format using replay image and mask data.")
    parser.add_argument(
        "roots",
        nargs="*",
        default=["dof_batch_results"],
        help="Result roots or individual case dirs to process. Defaults to dof_batch_results.",
    )
    parser.add_argument(
        "--replay-root",
        default="dof_match/teach_dof_replay",
        help="Replay artifact root used to reconstruct the after-mask panel.",
    )
    args = parser.parse_args()

    replay_root = Path(args.replay_root).expanduser()
    total_updated = 0
    for raw_root in args.roots:
        root = Path(raw_root).expanduser()
        for case_dir in _iter_case_dirs(root):
            total_updated += _restore_case(case_dir, replay_root)
    print(f"Restored {total_updated} PNG(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
