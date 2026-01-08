import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from utils.drawing import draw_axes, draw_cube_on_image
from utils.pose_io import load_pose_json
from utils.scene_io import parse_distortion_dict, parse_image_path_dict, parse_intrinsics_dict


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Draw a 3D axis on an image using intrinsics + pose.")

    p.add_argument(
        "--dir",
        type=Path,
        default=Path("."),
        help="Folder to search for a PNG if neither --image nor JSON image path is provided.",
    )
    p.add_argument("--image", type=Path, default=None, help="Path to an input .png image.")
    p.add_argument("--out", type=Path, default=Path("axis_overlay.png"), help="Output image path.")

    p.add_argument(
        "--K",
        type=float,
        nargs=4,
        metavar=("fx", "fy", "cx", "cy"),
        required=False,
        help="Camera intrinsics as fx fy cx cy.",
    )
    p.add_argument(
        "--pose-json",
        type=Path,
        required=False,
        help="JSON file with pose (and optionally intrinsics). Pose can be at top-level or under key 'pose'.",
    )

    p.add_argument("--axis-length", type=float, default=0.1, help="Axis length in world units.")
    p.add_argument("--cube-size", type=float, default=0.1, help="Cube edge length in world units.")
    p.add_argument("--line-width", type=int, default=5, help="Axis line width in pixels.")

    p.add_argument(
        "--draw",
        choices=("axes", "cube", "both"),
        default="axes",
        help="What to draw on the image.",
    )

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.pose_json is None:
        raise SystemExit(
            "Missing --pose-json. Put image+intrinsics+pose in a JSON and pass it via --pose-json."
        )

    pose_data = json.loads(args.pose_json.read_text())
    if not isinstance(pose_data, dict):
        raise SystemExit("--pose-json must be a JSON object")

    image_from_json = parse_image_path_dict(pose_data)
    if args.image is not None:
        image_path = args.image
    elif image_from_json is not None:
        candidate = Path(image_from_json)
        image_path = candidate if candidate.is_absolute() else (args.pose_json.parent / candidate)
    else:
        raise SystemExit(
            "Missing image. Provide --image or include image path in --pose-json."
        )

    K_from_json = parse_intrinsics_dict(pose_data)
    if args.K is not None:
        fx, fy, cx, cy = args.K
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float)
    elif K_from_json is not None:
        K = K_from_json
    else:
        raise SystemExit(
            "Missing intrinsics. Provide --K fx fy cx cy or put intrinsics in --pose-json (fx/fy/cx/cy or K 3x3)."
        )
    distortion = parse_distortion_dict(pose_data)

    print("Intrinsics (K):")
    print(K)
    print("Distortion coefficients:")
    print(distortion)

    T_w2c = load_pose_json(args.pose_json)

    # Render 
    img = Image.open(image_path)
    out_img = img
    if args.draw in ("axes", "both"):
        out_img = draw_axes(
            image=out_img,
            K=K,
            T_w2c=T_w2c,
            axis_length=args.axis_length,
            line_width=args.line_width,
            dist_coeffs=distortion,
        )
    if args.draw in ("cube", "both"):
        out_img = draw_cube_on_image(
            image=out_img,
            K=K,
            T_w2c=T_w2c,
            cube_size=args.cube_size,
            line_width=args.line_width,
            dist_coeffs=distortion,
        )

    # Write output
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(args.out)

    print(f"Input : {image_path.resolve()}")
    print(f"Output: {args.out.resolve()}")


if __name__ == "__main__":
    main()
