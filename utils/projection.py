import numpy as np


def project_points(
    K: np.ndarray,
    T_w2c: np.ndarray,
    Pw: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> np.ndarray:
    """Project Nx3 world points to Nx2 pixel coordinates with optional distortion.

    In this project, the pose used for projection is always the world->camera
    transform (w2c), named T_w2c.

    Args:
        K: 3x3 camera intrinsic matrix
        T_w2c: 4x4 world-to-camera transform
        Pw: Nx3 world points
        dist_coeffs: Optional distortion coefficients [k1, k2, p1, p2, k3] (OpenCV order)

    Returns:
        Nx2 pixel coordinates (NaN for points behind camera)
    """
    if K.shape != (3, 3):
        raise ValueError(f"Expected K shape (3,3), got {K.shape}")
    if T_w2c.shape != (4, 4):
        raise ValueError(f"Expected T_w2c shape (4,4), got {T_w2c.shape}")
    if Pw.ndim != 2 or Pw.shape[1] != 3:
        raise ValueError(f"Expected Pw shape (N,3), got {Pw.shape}")

    N = Pw.shape[0]
    Pw_h = np.concatenate([Pw.astype(float), np.ones((N, 1), dtype=float)], axis=1)  # (N,4)
    Pc_h = (T_w2c @ Pw_h.T).T  # (N,4)
    Pc = Pc_h[:, :3]

    z = Pc[:, 2]
    valid = z > 1e-8

    # Normalize to get camera coordinates (x/z, y/z)
    x_norm = np.full(N, np.nan, dtype=float)
    y_norm = np.full(N, np.nan, dtype=float)
    x_norm[valid] = Pc[valid, 0] / Pc[valid, 2]
    y_norm[valid] = Pc[valid, 1] / Pc[valid, 2]

    # Apply distortion if provided
    if dist_coeffs is not None:
        dist_coeffs = np.asarray(dist_coeffs, dtype=float).flatten()
        if dist_coeffs.size >= 5:
            k1, k2, p1, p2, k3 = dist_coeffs[:5]
        elif dist_coeffs.size == 4:
            k1, k2, p1, p2 = dist_coeffs
            k3 = 0.0
        elif dist_coeffs.size == 2:
            k1, k2 = dist_coeffs
            p1 = p2 = k3 = 0.0
        else:
            raise ValueError(f"Expected dist_coeffs size 2, 4, or 5, got {dist_coeffs.size}")

        # Apply distortion to valid points
        r2 = x_norm[valid] ** 2 + y_norm[valid] ** 2
        r4 = r2 * r2
        r6 = r4 * r2

        # Radial distortion
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6

        # Tangential distortion
        x_tangential = 2.0 * p1 * x_norm[valid] * y_norm[valid] + p2 * (r2 + 2.0 * x_norm[valid] ** 2)
        y_tangential = p1 * (r2 + 2.0 * y_norm[valid] ** 2) + 2.0 * p2 * x_norm[valid] * y_norm[valid]

        # Apply distortion
        x_norm[valid] = x_norm[valid] * radial + x_tangential
        y_norm[valid] = y_norm[valid] * radial + y_tangential

    # Apply intrinsics
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    uv = np.full((N, 2), np.nan, dtype=float)
    uv[valid, 0] = fx * x_norm[valid] + cx
    uv[valid, 1] = fy * y_norm[valid] + cy

    return uv


def warn_if_all_behind_camera(name: str, uv: np.ndarray) -> None:
    """Print a warning if all projected points are invalid/behind camera."""
    if uv.size == 0:
        return
    if np.all(np.isnan(uv)):
        print(
            f"WARNING: '{name}' projection is fully invalid (all points behind camera). "
            "Check pose convention/direction and units (axis/cube size)."
        )
