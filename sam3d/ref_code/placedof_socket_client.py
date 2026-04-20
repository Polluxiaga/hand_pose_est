#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import socket
import time
from pathlib import Path
from typing import Any

from placedof_protocol import PICK_CONTEXT_LEN, parse_pick_context, recv_framed_json, send_framed_json


def _file_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _parse_float_list(text: str) -> list[float]:
    raw = str(text or "").strip()
    if not raw:
        return []
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _load_pick_context_source(path: str) -> list[float]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if isinstance(raw.get("qwen_pick"), dict) and isinstance(raw["qwen_pick"].get("data"), list):
            return [float(v) for v in raw["qwen_pick"]["data"]]
        if isinstance(raw.get("results"), dict) and isinstance(raw["results"].get("pick"), dict):
            data = raw["results"]["pick"].get("data") or []
            return [float(v) for v in data]
        if isinstance(raw.get("data"), list) and len(raw["data"]) >= PICK_CONTEXT_LEN:
            return [float(v) for v in raw["data"][:PICK_CONTEXT_LEN]]
    if isinstance(raw, list):
        return [float(v) for v in raw]
    raise ValueError(f"Cannot parse pick context from {path}")


def call_placedof_socket_service(host: str, port: int, payload: dict[str, Any], timeout_s: float = 180.0) -> dict[str, Any]:
    with socket.create_connection((host, int(port)), timeout=float(timeout_s)) as sock:
        send_framed_json(sock, payload)
        return recv_framed_json(sock)


def main() -> None:
    parser = argparse.ArgumentParser(description="Client for placedof socket service")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5688)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--request-id", type=str, default="")
    parser.add_argument("--arm-id", type=int, default=0)
    parser.add_argument("--image-left", type=str, required=True)
    parser.add_argument("--image-right", type=str, required=True)
    parser.add_argument("--topic-inputs", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="placedof_socket_result.json")
    parser.add_argument("--log-mode", type=str, default="compact", choices=("full", "compact", "none"))
    parser.add_argument("--s2m2-api-url", type=str, default="")
    parser.add_argument("--sam3-url", type=str, default="")
    parser.add_argument("--target-kind-hint", type=str, default="")
    parser.add_argument("--pick-prompt", type=str, default="")
    parser.add_argument("--pick-label", type=str, default="")
    parser.add_argument("--pick-gesture", type=str, default="")
    parser.add_argument("--pick-mask", type=str, default="")
    parser.add_argument("--pick-offset", type=str, default="")
    parser.add_argument("--pick-shift-prop", type=float, default=None)
    parser.add_argument("--place-prompt", type=str, default="")
    parser.add_argument("--place-label", type=str, default="")
    parser.add_argument("--place-gesture", type=str, default="")
    parser.add_argument("--place-mask", type=str, default="")
    parser.add_argument("--place-offset", type=str, default="")
    parser.add_argument("--place-shift-prop", type=float, default=None)
    parser.add_argument("--pick-context-json", type=str, default="")
    args = parser.parse_args()

    topic_inputs_text = Path(args.topic_inputs).read_text(encoding="utf-8")
    payload: dict[str, Any] = {
        "request_id": args.request_id or f"client_{int(time.time() * 1000)}",
        "image_left_b64": _file_to_b64(args.image_left),
        "image_right_b64": _file_to_b64(args.image_right),
        "topic_inputs_text": topic_inputs_text,
        "arm_id": int(args.arm_id),
        "log_mode": str(args.log_mode),
    }
    if args.s2m2_api_url:
        payload["s2m2_api_url"] = str(args.s2m2_api_url).strip()
    if args.sam3_url:
        payload["sam3_url"] = str(args.sam3_url).strip()
    if args.target_kind_hint:
        payload["target_kind_hint"] = str(args.target_kind_hint).strip()

    def _build_prompt(label: str, gesture: str, mask: str) -> str:
        prompt = str(label or "").strip()
        if not prompt:
            return ""
        if str(gesture or "").strip():
            prompt += f":{str(gesture).strip()}"
        if str(mask or "").strip():
            prompt += f"@{str(mask).strip()}"
        return prompt

    if args.pick_prompt or args.pick_label:
        payload["pick_prompt"] = str(args.pick_prompt).strip() or _build_prompt(args.pick_label, args.pick_gesture, args.pick_mask)
        pick_data = _parse_float_list(args.pick_offset)
        if args.pick_shift_prop is not None:
            pick_data = pick_data[:3] + [float(args.pick_shift_prop)]
        payload["pick_data"] = pick_data

    if args.place_prompt or args.place_label:
        payload["place_prompt"] = str(args.place_prompt).strip() or _build_prompt(args.place_label, args.place_gesture, args.place_mask)
        place_data = _parse_float_list(args.place_offset)
        if args.place_shift_prop is not None:
            place_data = place_data[:3] + [float(args.place_shift_prop)]
        payload["place_data"] = place_data

    if args.pick_context_json:
        payload["pick_context_flat"] = _load_pick_context_source(args.pick_context_json)

    result = call_placedof_socket_service(host=str(args.host), port=int(args.port), payload=payload, timeout_s=float(args.timeout))
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ok={result.get('ok')}")
    print(f"error={result.get('error', '')}")
    print(f"request_id={result.get('request_id', '')}")
    if result.get("qwen_pick") is not None:
        pick_data = [float(v) for v in (result["qwen_pick"].get("data") or [])]
        print(f"pick_data_len={len(pick_data)}")
        if len(pick_data) >= PICK_CONTEXT_LEN:
            ctx = parse_pick_context(pick_data[:PICK_CONTEXT_LEN])
            print(f"pick_size_xyz={ctx['size_xyz'].tolist()}")
    if result.get("qwen_place") is not None:
        print(f"place_data_len={len(result['qwen_place'].get('data') or [])}")
    print(f"log_dir={result.get('log_dir', '')}")
    print(f"saved={args.output_json}")


if __name__ == "__main__":
    main()
