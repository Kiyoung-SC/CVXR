from __future__ import annotations

from typing import Optional

import numpy as np


def parse_intrinsics_dict(data: dict) -> Optional[np.ndarray]:
    """Parse camera intrinsics from a dict.

    Accepted formats:
    - {"fx":..,"fy":..,"cx":..,"cy":..}
    - {"intrinsics": {"fx":..,"fy":..,"cx":..,"cy":..}}
    - {"K": [[...3x3...]]}
    - {"camera_matrix": [[...3x3...]]}
    """

    def from_fxfy(data2: dict) -> Optional[np.ndarray]:
        if all(k in data2 for k in ("fx", "fy", "cx", "cy")):
            fx = float(data2["fx"])
            fy = float(data2["fy"])
            cx = float(data2["cx"])
            cy = float(data2["cy"])
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
        return None

    K = from_fxfy(data)
    if K is not None:
        return K

    if "intrinsics" in data and isinstance(data["intrinsics"], dict):
        K = from_fxfy(data["intrinsics"])
        if K is not None:
            return K

    for key in ("K", "camera_matrix"):
        if key in data:
            K_arr = np.array(data[key], dtype=float)
            if K_arr.shape != (3, 3):
                raise ValueError(f"Expected '{key}' shape (3,3), got {K_arr.shape}")
            return K_arr

    return None


def parse_distortion_dict(data: dict) -> Optional[np.ndarray]:
    """Parse distortion coefficients from a dict.

    Accepted formats:
    - {"k1":..,"k2":..,"k3":..,"p1":..,"p2":..}
    - {"distortion": {"k1":..,"k2":..,"k3":..,"p1":..,"p2":..}}
    - {"distortion_coeffs": [k1, k2, p1, p2, k3]}
    - {"dist_coeffs": [k1, k2, p1, p2, k3]}
    
    Returns array in OpenCV order: [k1, k2, p1, p2, k3]
    """

    def from_individual(data2: dict) -> Optional[np.ndarray]:
        if all(k in data2 for k in ("k1", "k2", "p1", "p2")):
            k1 = float(data2["k1"])
            k2 = float(data2["k2"])
            p1 = float(data2["p1"])
            p2 = float(data2["p2"])
            k3 = float(data2.get("k3", 0.0))
            return np.array([k1, k2, p1, p2, k3], dtype=float)
        return None

    dist = from_individual(data)
    if dist is not None:
        return dist

    if "distortion" in data and isinstance(data["distortion"], dict):
        dist = from_individual(data["distortion"])
        if dist is not None:
            return dist

    for key in ("distortion_coeffs", "dist_coeffs", "distortion", "dist"):
        if key in data and isinstance(data[key], (list, tuple, np.ndarray)):
            dist_arr = np.array(data[key], dtype=float).flatten()
            if dist_arr.size == 5:
                return dist_arr
            elif dist_arr.size == 4:
                # Assume [k1, k2, p1, p2], add k3=0
                return np.array([dist_arr[0], dist_arr[1], dist_arr[2], dist_arr[3], 0.0], dtype=float)
            elif dist_arr.size == 2:
                # Assume [k1, k2], add p1=p2=k3=0
                return np.array([dist_arr[0], dist_arr[1], 0.0, 0.0, 0.0], dtype=float)
            else:
                raise ValueError(f"Expected distortion coefficients size 2, 4, or 5, got {dist_arr.size}")

    return None


def parse_image_path_dict(data: dict) -> Optional[str]:
    """Parse an image filename/path from a dict.

    Accepted keys:
    - "image"
    - "image_path"
    - {"image": {"path": "..."}}
    """

    if "image" in data and isinstance(data["image"], str):
        return data["image"]
    if "image_path" in data and isinstance(data["image_path"], str):
        return data["image_path"]
    if "image" in data and isinstance(data["image"], dict):
        v = data["image"].get("path")
        if isinstance(v, str):
            return v
    return None
