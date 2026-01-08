import numpy as np


def invert_se3(T: np.ndarray) -> np.ndarray:
    """Invert a rigid 4x4 transform."""
    if T.shape != (4, 4):
        raise ValueError(f"Expected T shape (4,4), got {T.shape}")
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def quat_to_rotmat(q: np.ndarray, *, order: str) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.

    order:
      - 'xyzw' means q = [qx, qy, qz, qw]
      - 'wxyz' means q = [qw, qx, qy, qz]
    """

    q = np.asarray(q, dtype=float).reshape(4)
    order = order.lower()
    if order == "xyzw":
        x, y, z, w = q
    elif order == "wxyz":
        w, x, y, z = q
    else:
        raise ValueError("quat order must be 'xyzw' or 'wxyz'")

    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise ValueError("Quaternion norm is too small")
    w, x, y, z = w / n, x / n, y / n, z / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=float,
    )
