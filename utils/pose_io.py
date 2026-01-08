import json
from pathlib import Path

import numpy as np

from .geometry import invert_se3, quat_to_rotmat


def parse_pose_dict(data: dict) -> np.ndarray:
    """Parse pose and return T_w2c (world->camera).

    Supported inputs:
    - Matrix form:
      - {"T_w2c": [[4x4]]} or {"T_wc": [[4x4]]}
      - {"T_c2w": [[4x4]]} or {"T_cw": [[4x4]]} (will be inverted)
    - Pose form:
      - {"t": [...], "q": [...], "type": "c2w"|"w2c"}  (default: c2w)

    Notes:
    - "c2w" is common when pose is camera pose in world coordinates.
    """

    if "T_w2c" in data or "T_wc" in data:
        key = "T_w2c" if "T_w2c" in data else "T_wc"
        return np.array(data[key], dtype=float)

    if "T_c2w" in data or "T_cw" in data:
        key = "T_c2w" if "T_c2w" in data else "T_cw"
        return invert_se3(np.array(data[key], dtype=float))

    def normalize_pose_type(v: str) -> str:
        v = v.strip().lower()
        if v in ("c2w", "camera_to_world", "camera-to-world", "cw"):
            return "c2w"
        if v in ("w2c", "world_to_camera", "world-to-camera", "wc"):
            return "w2c"
        raise ValueError("pose JSON 'type' must be 'c2w' or 'w2c' (also accepts 'cw'/'wc')")

    if "t" in data and ("q" in data or all(k in data for k in ("qx", "qy", "qz", "qw"))):
        t = np.array(data["t"], dtype=float).reshape(3)

        if "q" in data:
            q = np.array(data["q"], dtype=float).reshape(4)
            quat_order = str(data.get("quat_order", "xyzw")).lower()
        else:
            q = np.array([data["qx"], data["qy"], data["qz"], data["qw"]], dtype=float)
            quat_order = "xyzw"

        R = quat_to_rotmat(q, order=quat_order)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t

        pose_type = normalize_pose_type(str(data.get("type", "c2w")))
        return invert_se3(T) if pose_type == "c2w" else T

    if "R" in data and "t" in data:
        R = np.array(data["R"], dtype=float)
        t = np.array(data["t"], dtype=float).reshape(3)
        T = np.eye(4, dtype=float)
        T[:3, :3] = R
        T[:3, 3] = t
        pose_type = normalize_pose_type(str(data.get("type", "c2w")))
        return invert_se3(T) if pose_type == "c2w" else T

    raise ValueError(
        "Unrecognized pose format. Provide T_w2c/T_wc or T_c2w/T_cw or (t,q) with optional type c2w/w2c."
    )


def load_pose_json(path: Path) -> np.ndarray:
    """Load pose from JSON and return T_w2c.

    Pose can be at top-level or under key 'pose'.
    """
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "pose" in data and isinstance(data["pose"], dict):
        return parse_pose_dict(data["pose"])
    if not isinstance(data, dict):
        raise ValueError("Pose JSON must be an object")
    return parse_pose_dict(data)
