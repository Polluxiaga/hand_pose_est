from __future__ import annotations

import json
import socket
from typing import Any

import numpy as np

PICK_XQUAT_LEN = 7
PICK_OBB_CENTER_LEN = 3
PICK_OBB_AXES_LEN = 9
PICK_OBB_SIZE_LEN = 3
PICK_GRASP_ORIGIN_LOCAL_LEN = 3
PICK_GRASP_AXES_LOCAL_LEN = 9
PICK_CONTEXT_LEN = (
    PICK_XQUAT_LEN
    + PICK_OBB_CENTER_LEN
    + PICK_OBB_AXES_LEN
    + PICK_OBB_SIZE_LEN
    + PICK_GRASP_ORIGIN_LOCAL_LEN
    + PICK_GRASP_AXES_LOCAL_LEN
)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks: list[bytes] = []
    got = 0
    while got < int(n):
        chunk = sock.recv(int(n) - got)
        if not chunk:
            raise RuntimeError("socket closed")
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def send_framed_json(sock: socket.socket, obj: dict[str, Any]) -> None:
    body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    sock.sendall(len(body).to_bytes(4, "big") + body)


def recv_framed_json(sock: socket.socket) -> dict[str, Any]:
    header = recv_exact(sock, 4)
    payload_len = int.from_bytes(header, "big")
    if payload_len <= 0:
        raise RuntimeError("invalid response length")
    payload = recv_exact(sock, payload_len)
    obj = json.loads(payload.decode("utf-8"))
    if not isinstance(obj, dict):
        raise RuntimeError("response is not a json object")
    return obj


def flatten_pick_context(
    pick_xquat: np.ndarray | list[float],
    obb_center_base: np.ndarray | list[float],
    obb_axes_base: np.ndarray | list[list[float]] | list[float],
    size_xyz: np.ndarray | list[float],
    grasp_origin_local: np.ndarray | list[float],
    grasp_axes_local: np.ndarray | list[list[float]] | list[float],
) -> list[float]:
    pick = np.asarray(pick_xquat, dtype=np.float32).reshape(PICK_XQUAT_LEN)
    center = np.asarray(obb_center_base, dtype=np.float32).reshape(PICK_OBB_CENTER_LEN)
    axes = np.asarray(obb_axes_base, dtype=np.float32).reshape(3, 3)
    size = np.asarray(size_xyz, dtype=np.float32).reshape(PICK_OBB_SIZE_LEN)
    origin_local = np.asarray(grasp_origin_local, dtype=np.float32).reshape(PICK_GRASP_ORIGIN_LOCAL_LEN)
    axes_local = np.asarray(grasp_axes_local, dtype=np.float32).reshape(3, 3)
    vals = np.concatenate(
        [
            pick,
            center,
            axes.reshape(-1),
            size,
            origin_local,
            axes_local.reshape(-1),
        ],
        axis=0,
    )
    return [float(v) for v in vals.tolist()]


def parse_pick_context(flat_data: list[float] | np.ndarray) -> dict[str, np.ndarray]:
    vals = np.asarray(flat_data, dtype=np.float32).reshape(-1)
    if vals.shape[0] < PICK_CONTEXT_LEN:
        raise ValueError(
            f"pick context expects at least {PICK_CONTEXT_LEN} floats, got {int(vals.shape[0])}"
        )
    return {
        "pick_xquat": vals[:PICK_XQUAT_LEN].astype(np.float32, copy=True),
        "obb_center_base": vals[PICK_XQUAT_LEN : PICK_XQUAT_LEN + PICK_OBB_CENTER_LEN].astype(np.float32, copy=True),
        "obb_axes_base": vals[
            PICK_XQUAT_LEN + PICK_OBB_CENTER_LEN : PICK_XQUAT_LEN + PICK_OBB_CENTER_LEN + PICK_OBB_AXES_LEN
        ].reshape(3, 3).astype(np.float32, copy=True),
        "size_xyz": vals[
            PICK_XQUAT_LEN + PICK_OBB_CENTER_LEN + PICK_OBB_AXES_LEN :
            PICK_XQUAT_LEN + PICK_OBB_CENTER_LEN + PICK_OBB_AXES_LEN + PICK_OBB_SIZE_LEN
        ].astype(np.float32, copy=True),
        "grasp_origin_local": vals[
            PICK_XQUAT_LEN + PICK_OBB_CENTER_LEN + PICK_OBB_AXES_LEN + PICK_OBB_SIZE_LEN :
            PICK_XQUAT_LEN + PICK_OBB_CENTER_LEN + PICK_OBB_AXES_LEN + PICK_OBB_SIZE_LEN + PICK_GRASP_ORIGIN_LOCAL_LEN
        ].astype(np.float32, copy=True),
        "grasp_axes_local": vals[
            PICK_XQUAT_LEN + PICK_OBB_CENTER_LEN + PICK_OBB_AXES_LEN + PICK_OBB_SIZE_LEN + PICK_GRASP_ORIGIN_LOCAL_LEN :
            PICK_CONTEXT_LEN
        ].reshape(3, 3).astype(np.float32, copy=True),
    }


def pick_context_to_jsonable(flat_data: list[float] | np.ndarray) -> dict[str, Any]:
    ctx = parse_pick_context(flat_data)
    return {
        "pick_xquat": [float(v) for v in ctx["pick_xquat"].tolist()],
        "obb_center_base": [float(v) for v in ctx["obb_center_base"].tolist()],
        "obb_axes_base": np.asarray(ctx["obb_axes_base"], dtype=np.float32).tolist(),
        "size_xyz": [float(v) for v in ctx["size_xyz"].tolist()],
        "grasp_origin_local": [float(v) for v in ctx["grasp_origin_local"].tolist()],
        "grasp_axes_local": np.asarray(ctx["grasp_axes_local"], dtype=np.float32).tolist(),
    }
