from __future__ import annotations

import base64
import io
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_K1_ROS_ROOT = _REPO_ROOT / "k1-ros"
if str(_K1_ROS_ROOT) not in sys.path:
    sys.path.insert(0, str(_K1_ROS_ROOT))

from knowin_perception.visual_processor import VisualProcessorNode
from knowin_reasoner.reasoner_node import ReasonerNode
from knowin_reasoner.services import EnvStore, ServiceBase, ServiceError, ServiceMeta

from placedof_protocol import PICK_CONTEXT_LEN, pick_context_to_jsonable, recv_framed_json, send_framed_json

DEFAULT_HOST = os.getenv("PLACEDOF_SOCKET_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("PLACEDOF_SOCKET_PORT", "5688"))
DEFAULT_TIMEOUT_S = float(os.getenv("PLACEDOF_SOCKET_TIMEOUT_S", "180.0"))
DEFAULT_FRAME_TIMEOUT_S = float(os.getenv("PLACEDOF_SOCKET_FRAME_TIMEOUT_S", "20.0"))


def _image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _capture_stereo_images(visual: VisualProcessorNode, timeout: float = DEFAULT_FRAME_TIMEOUT_S) -> tuple[Image.Image, Image.Image]:
    t0 = time.time()
    while visual.is_frame_timeout() or visual.last_left_image is None or visual.last_right_image is None:
        if time.time() - t0 > timeout:
            raise ServiceError(f"Save stereo frames timeout in {timeout} seconds")
        visual.get_logger().error("Save stereo frames failed! retrying...")
        time.sleep(0.5)
    try:
        left_bgr = visual.process_bgr_msg(visual.last_left_image)
        right_bgr = visual.process_bgr_msg(visual.last_right_image)
    except Exception as exc:
        raise ServiceError(f"Failed to convert stereo frames: {exc}") from exc
    left_rgb = Image.fromarray(left_bgr[:, :, ::-1]).convert("RGB")
    right_rgb = Image.fromarray(right_bgr[:, :, ::-1]).convert("RGB")
    return left_rgb, right_rgb


def _build_topic_inputs_text(visual: VisualProcessorNode, arm_id: int) -> str:
    if visual.intrinsics is None:
        raise ServiceError("Camera intrinsics are unavailable")
    if visual.namespace == "head":
        transform = visual.head_base_transforms[int(arm_id)]
    else:
        transform = visual.camera_base_transform
    if transform is None:
        raise ServiceError("Camera base transform is unavailable")
    fx = float(visual.intrinsics.fx)
    fy = float(visual.intrinsics.fy)
    cx = float(visual.intrinsics.cx)
    cy = float(visual.intrinsics.cy)
    baseline = float(visual.intrinsics.baseline_str)
    tx = fx * baseline * 1000.0
    p_values = [fx, 0.0, cx, tx, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    lines = [str(getattr(visual, "intrinsics_topic", "/camera_info")), "p:"]
    lines.extend(f"- {value}" for value in p_values)
    lines.append("")
    lines.append("T_cam_to_base:")
    lines.append(json.dumps({"rotation": transform.rotation.tolist(), "shift": transform.shift.tolist()}))
    return "\n".join(lines)


def _split_target_data(text: list[str], data: list[float]) -> list[list[float]]:
    vals = [float(v) for v in data]
    if not vals:
        return [[] for _ in text]
    if len(vals) in (3, 4):
        return [list(vals) for _ in text]
    if len(vals) == len(text) * 3:
        return [vals[i * 3 : (i + 1) * 3] for i in range(len(text))]
    if len(vals) == len(text) * 4:
        return [vals[i * 4 : (i + 1) * 4] for i in range(len(text))]
    raise ServiceError(
        "text/data compatibility expects len(data) in {0,3,4,len(text)*3,len(text)*4}, "
        f"got len(text)={len(text)}, len(data)={len(vals)}"
    )


class _BaseSocketService(ServiceBase):
    def __init__(self, parent: ReasonerNode, env_store: EnvStore):
        super().__init__(parent, env_store)
        self.visual = env_store.get_singleton(VisualProcessorNode)
        self._host = os.getenv("PLACEDOF_SOCKET_HOST", DEFAULT_HOST)
        self._port = int(os.getenv("PLACEDOF_SOCKET_PORT", str(DEFAULT_PORT)))
        self._timeout_s = float(os.getenv("PLACEDOF_SOCKET_TIMEOUT_S", str(DEFAULT_TIMEOUT_S)))
        self._frame_timeout_s = float(os.getenv("PLACEDOF_SOCKET_FRAME_TIMEOUT_S", str(DEFAULT_FRAME_TIMEOUT_S)))

    def _build_common_payload(self, arm_id: int) -> dict[str, Any]:
        left_rgb, right_rgb = _capture_stereo_images(self.visual, timeout=self._frame_timeout_s)
        topic_inputs_text = _build_topic_inputs_text(self.visual, arm_id=arm_id)
        return {
            "request_id": f"reasoner_{int(time.time() * 1000)}",
            "image_left_b64": _image_to_b64(left_rgb),
            "image_right_b64": _image_to_b64(right_rgb),
            "topic_inputs_text": topic_inputs_text,
            "arm_id": int(arm_id),
            "log_mode": "compact",
        }

    def _call_socket(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with socket.create_connection((self._host, int(self._port)), timeout=float(self._timeout_s)) as sock:
                send_framed_json(sock, payload)
                result = recv_framed_json(sock)
        except Exception as exc:
            raise ServiceError(f"placedof socket call failed: {exc}") from exc
        if not isinstance(result, dict):
            raise ServiceError("Invalid placedof response type")
        if not result.get("ok", False):
            raise ServiceError(f"placedof error: {result.get('error', 'unknown')}")
        return result


@ServiceMeta(service_name="placedof_pick_dof")
class PlaceDofPickService(_BaseSocketService):
    """
    Request:
      text: [pick_prompt]
      data: [] / [dx,dy,dz] / [dx,dy,dz,shift]

    Response:
      text: [pick_label]
      data: flattened pick context
        - [:7]      pick xquat, backward-compatible with qwen_sam3_s2m2
        - [7:34]    fitted pick OBB + local grasp context for placedof place planning
    """

    def __call__(self, text: list[str], data: list[float], arm_id: int = 0) -> tuple[list[str], list[float]]:
        prompts = [str(item).strip() for item in text if str(item).strip()]
        if len(prompts) != 1:
            raise ServiceError("placedof_pick_dof requires exactly one pick prompt")
        if data and len(data) not in (3, 4):
            raise ServiceError("placedof_pick_dof expects []/[dx,dy,dz]/[dx,dy,dz,shift]")
        payload = self._build_common_payload(arm_id=int(arm_id))
        payload["pick_prompt"] = prompts[0]
        payload["pick_data"] = [float(v) for v in data]
        result = self._call_socket(payload)
        qwen_pick = result.get("qwen_pick")
        if not isinstance(qwen_pick, dict):
            raise ServiceError("placedof_pick_dof returned invalid qwen_pick")
        out_text = [str(v) for v in qwen_pick.get("text") or []]
        out_data = [float(v) for v in qwen_pick.get("data") or []]
        if len(out_data) < PICK_CONTEXT_LEN:
            raise ServiceError("placedof_pick_dof returned incomplete pick context")
        return out_text, out_data


@ServiceMeta(service_name="placedof_place_dof")
class PlaceDofPlaceService(_BaseSocketService):
    """
    Request:
      text: [place_prompt]
      data:
        - first 34 floats: pick context returned by placedof_pick_dof
        - trailing []/[dx,dy,dz]/[dx,dy,dz,shift]: optional place bias payload kept for compatibility

    Response:
      text: [place_label]
      data: place grasp xquat (7)
    """

    def __call__(self, text: list[str], data: list[float], arm_id: int = 0) -> tuple[list[str], list[float]]:
        prompts = [str(item).strip() for item in text if str(item).strip()]
        if len(prompts) != 1:
            raise ServiceError("placedof_place_dof requires exactly one place prompt")
        vals = [float(v) for v in data]
        if len(vals) < PICK_CONTEXT_LEN:
            raise ServiceError(
                f"placedof_place_dof requires pick context first; got {len(vals)} floats, need at least {PICK_CONTEXT_LEN}"
            )
        pick_context_flat = vals[:PICK_CONTEXT_LEN]
        trailing = vals[PICK_CONTEXT_LEN:]
        if trailing and len(trailing) not in (3, 4):
            raise ServiceError("placedof_place_dof trailing data must be []/[dx,dy,dz]/[dx,dy,dz,shift]")
        pick_context = pick_context_to_jsonable(pick_context_flat)
        payload = self._build_common_payload(arm_id=int(arm_id))
        payload["place_prompt"] = prompts[0]
        payload["place_data"] = [float(v) for v in trailing]
        payload["pick_context"] = pick_context
        result = self._call_socket(payload)
        qwen_place = result.get("qwen_place")
        if not isinstance(qwen_place, dict):
            raise ServiceError("placedof_place_dof returned invalid qwen_place")
        out_text = [str(v) for v in qwen_place.get("text") or []]
        out_data = [float(v) for v in qwen_place.get("data") or []]
        if len(out_data) < 7:
            raise ServiceError("placedof_place_dof returned invalid place xquat")
        return out_text, out_data[:7]


@ServiceMeta(service_name="placedof_pickplace_dof")
class PlaceDofPickPlaceService(_BaseSocketService):
    """
    Request:
      text: [pick_prompt, place_prompt]
      data: [] / shared [dx,dy,dz(,shift)] / concatenated per target

    Response:
      text: [pick_label, place_label]
      data:
        - pick full context first (34)
        - place xquat after that (7)
    """

    def __call__(self, text: list[str], data: list[float], arm_id: int = 0) -> tuple[list[str], list[float]]:
        prompts = [str(item).strip() for item in text if str(item).strip()]
        if not prompts:
            raise ServiceError("placedof_pickplace_dof requires prompts")
        if len(prompts) > 2:
            raise ServiceError("placedof_pickplace_dof supports at most two prompts: [pick, place]")
        grouped_data = _split_target_data(prompts, [float(v) for v in data])
        payload = self._build_common_payload(arm_id=int(arm_id))
        if len(prompts) >= 1:
            payload["pick_prompt"] = prompts[0]
            payload["pick_data"] = grouped_data[0]
        if len(prompts) >= 2:
            payload["place_prompt"] = prompts[1]
            payload["place_data"] = grouped_data[1]
        result = self._call_socket(payload)
        qwen_pick = result.get("qwen_pick")
        qwen_place = result.get("qwen_place")
        out_text: list[str] = []
        out_data: list[float] = []
        if isinstance(qwen_pick, dict):
            out_text.extend([str(v) for v in qwen_pick.get("text") or []])
            out_data.extend([float(v) for v in qwen_pick.get("data") or []])
        if isinstance(qwen_place, dict):
            out_text.extend([str(v) for v in qwen_place.get("text") or []])
            out_data.extend([float(v) for v in qwen_place.get("data") or []])
        if not out_data:
            raise ServiceError("placedof_pickplace_dof returned empty data")
        return out_text, out_data
