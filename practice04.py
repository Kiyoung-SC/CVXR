#!/usr/bin/env python3
"""Detect ArUco marker, extract pose, and render 3D axis."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from utils.drawing import draw_axes
from utils.geometry import invert_se3
from utils.scene_io import parse_distortion_dict, parse_intrinsics_dict


def detect_aruco_markers(image: np.ndarray, dict_type: str = "4X4_50"):
    """Detect ArUco markers in an image.
    
    Args:
        image: Input image (BGR format from cv2)
        dict_type: ArUco dictionary type
    
    Returns:
        corners: List of detected marker corners
        ids: List of detected marker IDs
        rejected: List of rejected candidates
    """
    aruco_dicts = {
        "4X4_50": cv2.aruco.DICT_4X4_50,
        "4X4_100": cv2.aruco.DICT_4X4_100,
        "5X5_50": cv2.aruco.DICT_5X5_50,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "6X6_50": cv2.aruco.DICT_6X6_50,
        "6X6_100": cv2.aruco.DICT_6X6_100,
    }
    
    if dict_type not in aruco_dicts:
        raise ValueError(f"Unknown ArUco dictionary: {dict_type}")
    
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dicts[dict_type])
    parameters = cv2.aruco.DetectorParameters()
    
    # Make detection more lenient
    parameters.adaptiveThreshConstant = 7
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 4.0
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 0
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Use legacy detectMarkers function which is more reliable
    corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    
    return corners, ids, rejected


def estimate_pose_single_marker(
    corners: np.ndarray,
    marker_size: float,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
):
    """Estimate pose of a single ArUco marker.
    
    Args:
        corners: Marker corners (4x2 array)
        marker_size: Real-world marker size in meters
        K: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    
    Returns:
        rvec: Rotation vector
        tvec: Translation vector
        T_c2m: 4x4 transformation from camera to marker
    """
    if dist_coeffs is None:
        dist_coeffs = np.zeros(5)
    
    # Define 3D marker corners in marker coordinate system
    # ArUco markers have corners at [-size/2, size/2] in X and Y
    half_size = marker_size / 2.0
    obj_points = np.array([
        [-half_size, half_size, 0],   # Top-left
        [half_size, half_size, 0],    # Top-right
        [half_size, -half_size, 0],   # Bottom-right
        [-half_size, -half_size, 0],  # Bottom-left
    ], dtype=np.float32)
    
    # Estimate pose
    success, rvec, tvec = cv2.solvePnP(
        obj_points, corners, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    
    if not success:
        return None, None, None
    
    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)
    
    # Build 4x4 transformation matrix (camera to marker)
    T_c2m = np.eye(4, dtype=float)
    T_c2m[:3, :3] = R
    T_c2m[:3, 3] = tvec.flatten()
    
    return rvec, tvec, T_c2m


def parse_args():
    p = argparse.ArgumentParser(description="Detect ArUco marker and render 3D axis.")
    
    p.add_argument("--image", type=Path, help="Input image path (optional if JSON has image_path)")
    p.add_argument("--output", type=Path, default=Path("output_detected.png"), help="Output image path")
    p.add_argument("--json", type=Path, help="JSON with camera intrinsics (optional)")
    
    # Camera intrinsics (if not provided via JSON)
    p.add_argument("--fx", type=float, default=500.0, help="Focal length x")
    p.add_argument("--fy", type=float, default=500.0, help="Focal length y")
    p.add_argument("--cx", type=float, default=320.0, help="Principal point x")
    p.add_argument("--cy", type=float, default=240.0, help="Principal point y")
    
    # Marker parameters
    p.add_argument("--marker-size", type=float, default=0.4, help="Real-world marker size in meters")
    p.add_argument("--marker-dict", type=str, default="4X4_50", help="ArUco dictionary type")
    
    # Rendering parameters
    p.add_argument("--axis-length", type=float, default=0.2, help="Axis length in world units")
    p.add_argument("--line-width", type=int, default=3, help="Axis line width in pixels")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    # Load camera intrinsics and optionally image path from JSON
    if args.json:
        json_data = json.loads(args.json.read_text())
        K = parse_intrinsics_dict(json_data)
        dist_coeffs = parse_distortion_dict(json_data)
        if K is None:
            raise ValueError("Could not parse intrinsics from JSON")
        
        # Try to get image path from JSON if not provided as argument
        if args.image is None:
            image_path = json_data.get("image_path")
            if image_path:
                args.image = Path(image_path)
    else:
        K = np.array([[args.fx, 0.0, args.cx], [0.0, args.fy, args.cy], [0.0, 0.0, 1.0]], dtype=float)
        dist_coeffs = None
    
    # Validate image path
    if args.image is None:
        raise ValueError("Image path must be provided via --image or in JSON file")
    
    print("Camera intrinsics K:")
    print(K)
    if dist_coeffs is not None:
        print("Distortion coefficients:")
        print(dist_coeffs)
    
    # Load image
    img_cv = cv2.imread(str(args.image))
    if img_cv is None:
        raise FileNotFoundError(f"Could not load image: {args.image}")
    
    print(f"\nLoaded image: {args.image} ({img_cv.shape[1]}x{img_cv.shape[0]})")
    
    # Detect ArUco markers
    corners, ids, rejected = detect_aruco_markers(img_cv, dict_type=args.marker_dict)
    
    if ids is None or len(ids) == 0:
        print("No ArUco markers detected!")
        return
    
    print(f"\nDetected {len(ids)} marker(s): {ids.flatten()}")
    
    # Process first detected marker
    marker_corners = corners[0].reshape(-1, 2)
    marker_id = ids[0][0]
    
    print(f"\nProcessing marker ID {marker_id}")
    print(f"Detected corners (pixels):\n{marker_corners}")
    
    # Estimate pose
    rvec, tvec, T_c2m = estimate_pose_single_marker(
        marker_corners, args.marker_size, K, dist_coeffs
    )
    
    if T_c2m is None:
        print("Failed to estimate marker pose!")
        return
    
    print(f"\nEstimated pose (camera to marker):")
    print(f"Rotation vector (rvec):\n{rvec.flatten()}")
    print(f"Translation vector (tvec):\n{tvec.flatten()}")
    print(f"\nTransformation matrix T_c2m (camera to marker):")
    print(T_c2m)
    
    # The marker coordinate system has origin at marker center, Z pointing out of marker
    # To render axes on the marker, we need T_w2c where world = marker
    # T_w2c = inverse(T_c2m) since world frame = marker frame
    T_m2c = invert_se3(T_c2m)
    print(f"\nTransformation matrix T_m2c (marker to camera, i.e., T_w2c for rendering):")
    print(T_m2c)
    
    # Convert to PIL and render axes
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    
    # Draw 3D axes at marker origin
    img_pil = draw_axes(
        image=img_pil,
        K=K,
        T_w2c=T_m2c,
        axis_length=args.axis_length,
        line_width=args.line_width,
        dist_coeffs=dist_coeffs,
    )
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    img_pil.save(args.output)
    
    print(f"\nSaved output: {args.output}")


if __name__ == "__main__":
    main()
