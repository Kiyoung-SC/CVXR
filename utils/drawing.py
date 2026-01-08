from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .projection import project_points, warn_if_all_behind_camera

def draw_axes(
    image: Image.Image,
    K: np.ndarray,
    T_w2c: np.ndarray,
    axis_length: float,
    line_width: int,
    dist_coeffs: np.ndarray | None = None,
) -> Image.Image:
    """Draw XYZ axes (X=red, Y=green, Z=blue) projected onto the image."""

    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)

    Pw = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_length, 0.0, 0.0],
            [0.0, axis_length, 0.0],
            [0.0, 0.0, axis_length],
        ],
        dtype=float,
    )

    uv = project_points(K, T_w2c, Pw, dist_coeffs)
    warn_if_all_behind_camera("axes", uv)
    o, x, y, z = uv

    def to_xy(pt: np.ndarray) -> tuple[float, float] | None:
        if np.any(np.isnan(pt)):
            return None
        return float(pt[0]), float(pt[1])

    o_xy = to_xy(o)
    x_xy = to_xy(x)
    y_xy = to_xy(y)
    z_xy = to_xy(z)

    if o_xy and x_xy:
        draw.line([o_xy, x_xy], fill=(255, 0, 0), width=line_width)
    if o_xy and y_xy:
        draw.line([o_xy, y_xy], fill=(0, 255, 0), width=line_width)
    if o_xy and z_xy:
        draw.line([o_xy, z_xy], fill=(0, 0, 255), width=line_width)

    return img


def draw_cube_on_image(
    image: Image.Image,
    K: np.ndarray,
    T_w2c: np.ndarray,
    cube_size: float,
    line_width: int,
    color: tuple[int, int, int] = (255, 255, 0),
    dist_coeffs: np.ndarray | None = None,
) -> Image.Image:
    """Draw a wireframe cube projected onto the image.

    The cube is defined in world coordinates with one corner at the origin and
    edges along +X, +Y, +Z, each of length cube_size.
    """

    img = image.convert("RGB")
    draw = ImageDraw.Draw(img)

    s = float(cube_size)
    Pw = np.array(
        [
            [0.0, 0.0, 0.0],
            [s, 0.0, 0.0],
            [s, s, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, s],
            [s, 0.0, s],
            [s, s, s],
            [0.0, s, s],
        ],
        dtype=float,
    )

    uv = project_points(K, T_w2c, Pw, dist_coeffs)
    warn_if_all_behind_camera("cube", uv)

    def to_xy(pt: np.ndarray) -> tuple[float, float] | None:
        if np.any(np.isnan(pt)):
            return None
        return float(pt[0]), float(pt[1])

    pts = [to_xy(uv[i]) for i in range(8)]

    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for a, b in edges:
        pa = pts[a]
        pb = pts[b]
        if pa is None or pb is None:
            continue
        draw.line([pa, pb], fill=color, width=line_width)

    return img
