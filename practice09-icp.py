#!/usr/bin/env python3
"""
Practice 09 ICP demo:
- Generate a car-like point cloud (src)
- Transform it to create dst
- Press 's' to run ICP step-by-step with visualization
"""

import argparse
import math
import time
import numpy as np
import open3d as o3d


def sample_box_surface(center, size, n_points):
    cx, cy, cz = center
    lx, ly, lz = size
    points = []
    faces = [
        (np.array([1, 0, 0]), lx / 2.0),
        (np.array([-1, 0, 0]), lx / 2.0),
        (np.array([0, 1, 0]), ly / 2.0),
        (np.array([0, -1, 0]), ly / 2.0),
        (np.array([0, 0, 1]), lz / 2.0),
        (np.array([0, 0, -1]), lz / 2.0),
    ]
    for _ in range(n_points):
        normal, half = faces[np.random.randint(0, len(faces))]
        u = np.random.uniform(-0.5, 0.5)
        v = np.random.uniform(-0.5, 0.5)
        if abs(normal[0]) == 1:
            x = cx + normal[0] * half
            y = cy + u * ly
            z = cz + v * lz
        elif abs(normal[1]) == 1:
            x = cx + u * lx
            y = cy + normal[1] * half
            z = cz + v * lz
        else:
            x = cx + u * lx
            y = cy + v * ly
            z = cz + normal[2] * half
        points.append([x, y, z])
    return np.array(points, dtype=float)


def sample_cylinder_surface(center, radius, width, n_points):
    cx, cy, cz = center
    points = []
    for _ in range(n_points):
        theta = np.random.uniform(0, 2 * math.pi)
        y = np.random.uniform(-width / 2.0, width / 2.0)
        x = cx + radius * math.cos(theta)
        z = cz + radius * math.sin(theta)
        points.append([x, cy + y, z])
    return np.array(points, dtype=float)


def generate_car_point_cloud(n_points=2000, seed=7, noise_scale=1.0):
    rng = np.random.default_rng(seed)
    np.random.seed(seed)

    body = sample_box_surface(center=(0.0, 0.0, 0.45), size=(4.2, 1.9, 0.9), n_points=int(n_points * 0.55))
    roof = sample_box_surface(center=(0.1, 0.0, 1.0), size=(2.2, 1.4, 0.5), n_points=int(n_points * 0.2))

    wheel_centers = [
        (-1.5, -0.9, 0.2),
        (-1.5, 0.9, 0.2),
        (1.5, -0.9, 0.2),
        (1.5, 0.9, 0.2),
    ]
    wheel_points = []
    for c in wheel_centers:
        wheel_points.append(sample_cylinder_surface(c, radius=0.35, width=0.25, n_points=int(n_points * 0.0625)))
    wheels = np.vstack(wheel_points)

    points = np.vstack([body, roof, wheels])

    jitter = rng.normal(scale=0.01 * noise_scale, size=points.shape)
    return points + jitter


def euler_to_rotation(rx, ry, rz):
    cx, cy, cz = math.cos(rx), math.cos(ry), math.cos(rz)
    sx, sy, sz = math.sin(rx), math.sin(ry), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_m @ ry_m @ rx_m


def rotation_angle(r):
    trace = float(np.trace(r))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    return math.acos(cos_theta)


def save_point_clouds(path, src_points, dst_points):
    np.savez(path, src=src_points, dst=dst_points)


def load_point_clouds(path):
    data = np.load(path)
    return np.array(data["src"], dtype=float), np.array(data["dst"], dtype=float)


def best_fit_transform(src, dst):
    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)
    src_centered = src - src_centroid
    dst_centered = dst - dst_centroid
    h = src_centered.T @ dst_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[2, :] *= -1
        r = vt.T @ u.T
    t = dst_centroid - r @ src_centroid
    return r, t


def icp_step(src_points, dst_points, dst_kdtree, max_dist):
    correspondences = []
    src_used = []
    for p in src_points:
        _, idx, dist2 = dst_kdtree.search_knn_vector_3d(p, 1)
        if not idx:
            continue
        dist = math.sqrt(dist2[0])
        if dist <= max_dist:
            correspondences.append(dst_points[idx[0]])
            src_used.append(p)
    if len(correspondences) < 3:
        return src_points, None, None, None, 0
    src_used = np.array(src_used, dtype=float)
    correspondences = np.array(correspondences, dtype=float)
    r, t = best_fit_transform(src_used, correspondences)
    transformed = (r @ src_points.T).T + t
    mean_error = float(np.mean(np.linalg.norm((r @ src_used.T).T + t - correspondences, axis=1)))
    return transformed, r, t, mean_error, len(correspondences)


def icp_step_point_to_plane(src_points, dst_points, dst_normals, dst_kdtree, max_dist):
    src_used = []
    dst_used = []
    n_used = []
    for p in src_points:
        _, idx, dist2 = dst_kdtree.search_knn_vector_3d(p, 1)
        if not idx:
            continue
        dist = math.sqrt(dist2[0])
        if dist <= max_dist:
            dst_used.append(dst_points[idx[0]])
            n_used.append(dst_normals[idx[0]])
            src_used.append(p)
    if len(dst_used) < 3:
        return src_points, None, None, None, 0

    src_used = np.array(src_used, dtype=float)
    dst_used = np.array(dst_used, dtype=float)
    n_used = np.array(n_used, dtype=float)

    a = np.zeros((len(src_used), 6), dtype=float)
    b = np.zeros(len(src_used), dtype=float)
    for i, (p, q, n) in enumerate(zip(src_used, dst_used, n_used)):
        cross = np.cross(p, n)
        a[i, :3] = cross
        a[i, 3:] = n
        b[i] = np.dot(n, q - p)

    x, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    rx, ry, rz, tx, ty, tz = x
    r_inc = euler_to_rotation(rx, ry, rz)
    t_inc = np.array([tx, ty, tz], dtype=float)
    transformed = (r_inc @ src_points.T).T + t_inc
    transformed_used = (r_inc @ src_used.T).T + t_inc
    residuals = np.abs(np.sum((transformed_used - dst_used) * n_used, axis=1))
    mean_error = float(np.mean(residuals))
    return transformed, r_inc, t_inc, mean_error, len(dst_used)


def icp_step_plane_to_plane(src_points, src_normals, dst_points, dst_normals, dst_kdtree, max_dist):
    src_used = []
    dst_used = []
    n_used = []
    for p, n_src in zip(src_points, src_normals):
        _, idx, dist2 = dst_kdtree.search_knn_vector_3d(p, 1)
        if not idx:
            continue
        dist = math.sqrt(dist2[0])
        if dist <= max_dist:
            n_dst = dst_normals[idx[0]]
            n_avg = n_src + n_dst
            norm = np.linalg.norm(n_avg)
            if norm < 1e-6:
                continue
            n_used.append(n_avg / norm)
            dst_used.append(dst_points[idx[0]])
            src_used.append(p)
    if len(dst_used) < 3:
        return src_points, src_normals, None, None, 0

    src_used = np.array(src_used, dtype=float)
    dst_used = np.array(dst_used, dtype=float)
    n_used = np.array(n_used, dtype=float)

    a = np.zeros((len(src_used), 6), dtype=float)
    b = np.zeros(len(src_used), dtype=float)
    for i, (p, q, n) in enumerate(zip(src_used, dst_used, n_used)):
        a[i, :3] = np.cross(p, n)
        a[i, 3:] = n
        b[i] = np.dot(n, q - p)

    x, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    rx, ry, rz, tx, ty, tz = x
    r_inc = euler_to_rotation(rx, ry, rz)
    t_inc = np.array([tx, ty, tz], dtype=float)
    transformed = (r_inc @ src_points.T).T + t_inc
    transformed_normals = (r_inc @ src_normals.T).T
    transformed_used = (r_inc @ src_used.T).T + t_inc
    residuals = np.abs(np.sum((transformed_used - dst_used) * n_used, axis=1))
    mean_error = float(np.mean(residuals))
    return transformed, transformed_normals, r_inc, t_inc, mean_error, len(dst_used)


def main():
    print("ICP demo - press 's' to start ICP, close the window to exit.")

    parser = argparse.ArgumentParser(description="ICP point cloud demo")
    parser.add_argument("--n-points", type=int, default=3000, help="Number of points in src cloud")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--src-noise", type=float, default=1.0,
                        help="Noise scale for src point cloud jitter (1.0 = default)")
    parser.add_argument("--dst-noise", type=float, default=1.0,
                        help="Noise scale for dst point cloud noise (1.0 = default)")
    parser.add_argument("--rot-deg", type=float, nargs=3, default=[-30.0, 5.0, 20.0],
                        metavar=("RX", "RY", "RZ"),
                        help="Rotation in degrees applied to src to make dst (XYZ order)")
    parser.add_argument("--trans", type=float, nargs=3, default=[0.5, -0.3, 0.15],
                        metavar=("TX", "TY", "TZ"),
                        help="Translation applied to src to make dst")
    parser.add_argument("--max-iterations", type=int, default=30, help="ICP iterations")
    parser.add_argument("--max-dist", type=float, default=0.5, help="ICP correspondence distance threshold")
    parser.add_argument("--icp-method", choices=["point2point", "point2plane", "plane2plane"], default="point2point",
                        help="ICP variant to use")
    parser.add_argument("--stop-rot", type=float, default=1e-3,
                        help="Stop if rotation update (rad) is below this threshold")
    parser.add_argument("--stop-trans", type=float, default=1e-3,
                        help="Stop if translation update is below this threshold")
    parser.add_argument("--io-path", default="icp_points.npz",
                        help="Path for saving/loading point sets (npz)")
    parser.add_argument("--step-delay", type=float, default=0.3, help="Seconds per ICP iteration")
    parser.add_argument("--point-size", type=float, default=5.0, help="Visualizer point size")
    args = parser.parse_args()

    src_points = generate_car_point_cloud(
        n_points=max(1, args.n_points),
        seed=args.seed,
        noise_scale=max(0.0, args.src_noise),
    )

    rx, ry, rz = args.rot_deg
    r_true = euler_to_rotation(math.radians(rx), math.radians(ry), math.radians(rz))
    t_true = np.array(args.trans, dtype=float)
    dst_points = (r_true @ src_points.T).T + t_true
    dst_points += np.random.normal(scale=0.005 * max(0.0, args.dst_noise), size=dst_points.shape)

    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_points)
    src_pcd.paint_uniform_color([0.9, 0.2, 0.2])

    dst_pcd = o3d.geometry.PointCloud()
    dst_pcd.points = o3d.utility.Vector3dVector(dst_points)
    dst_pcd.paint_uniform_color([0.2, 0.2, 0.9])

    max_iterations = max(1, args.max_iterations)
    max_dist = max(0.0, args.max_dist)
    step_delay = max(0.0, args.step_delay)
    stop_rot = max(0.0, args.stop_rot)
    stop_trans = max(0.0, args.stop_trans)
    running = {"flag": False}
    state = {
        "method": args.icp_method,
        "iter": 0,
        "initial_points": np.asarray(src_pcd.points).copy(),
        "initial_normals": None,
        "dst_kdtree": o3d.geometry.KDTreeFlann(dst_pcd),
    }

    def update_title(vis):
        title = f"ICP Demo [{state['method']}] iter {state['iter']}"
        if hasattr(vis, "set_window_title"):
            vis.set_window_title(title)
        else:
            print(title)

    def ensure_normals_for_method():
        if state["method"] in ("point2plane", "plane2plane") and not dst_pcd.has_normals():
            dst_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30)
            )
            dst_pcd.normalize_normals()
        if state["method"] == "plane2plane" and not src_pcd.has_normals():
            src_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30)
            )
            src_pcd.normalize_normals()
            state["initial_normals"] = np.asarray(src_pcd.normals).copy()

    def run_icp(vis):
        if running["flag"]:
            return False
        running["flag"] = True
        ensure_normals_for_method()
        current = np.asarray(src_pcd.points)
        current_normals = np.asarray(src_pcd.normals) if src_pcd.has_normals() else None
        state["iter"] = 0
        update_title(vis)
        for i in range(max_iterations):
            if state["method"] == "point2plane":
                current, r_inc, t_inc, mean_error, used = icp_step_point_to_plane(
                    current,
                    np.asarray(dst_pcd.points),
                    np.asarray(dst_pcd.normals),
                    state["dst_kdtree"],
                    max_dist,
                )
            elif state["method"] == "plane2plane":
                current, current_normals, r_inc, t_inc, mean_error, used = icp_step_plane_to_plane(
                    current,
                    current_normals,
                    np.asarray(dst_pcd.points),
                    np.asarray(dst_pcd.normals),
                    state["dst_kdtree"],
                    max_dist,
                )
            else:
                current, r_inc, t_inc, mean_error, used = icp_step(
                    current, np.asarray(dst_pcd.points), state["dst_kdtree"], max_dist
                )
            if mean_error is None:
                print(f"iter {i + 1:02d}: insufficient correspondences (threshold={max_dist})")
                break
            src_pcd.points = o3d.utility.Vector3dVector(current)
            if current_normals is not None:
                src_pcd.normals = o3d.utility.Vector3dVector(current_normals)
            vis.update_geometry(src_pcd)
            vis.poll_events()
            vis.update_renderer()
            state["iter"] = i + 1
            update_title(vis)
            rot_update = rotation_angle(r_inc)
            trans_update = float(np.linalg.norm(t_inc))
            print(
                f"iter {i + 1:02d}: mean error = {mean_error:.6f} (used={used}) "
                f"rot={rot_update:.6f} trans={trans_update:.6f}"
            )
            if rot_update <= stop_rot and trans_update <= stop_trans:
                print(f"stop: update below thresholds (rot<={stop_rot}, trans<={stop_trans})")
                break
            time.sleep(step_delay)
        running["flag"] = False
        return False

    def reset_view(vis):
        state["iter"] = 0
        src_pcd.points = o3d.utility.Vector3dVector(state["initial_points"])
        if state["initial_normals"] is not None:
            src_pcd.normals = o3d.utility.Vector3dVector(state["initial_normals"])
        elif src_pcd.has_normals():
            src_pcd.normals = o3d.utility.Vector3dVector()
        vis.update_geometry(src_pcd)
        vis.poll_events()
        vis.update_renderer()
        update_title(vis)
        return False

    def save_points(vis):
        save_point_clouds(args.io_path, np.asarray(src_pcd.points), np.asarray(dst_pcd.points))
        print(f"saved point sets to {args.io_path}")
        return False

    def load_points(vis):
        try:
            src_loaded, dst_loaded = load_point_clouds(args.io_path)
        except Exception as exc:
            print(f"failed to load {args.io_path}: {exc}")
            return False
        src_pcd.points = o3d.utility.Vector3dVector(src_loaded)
        dst_pcd.points = o3d.utility.Vector3dVector(dst_loaded)
        state["dst_kdtree"] = o3d.geometry.KDTreeFlann(dst_pcd)
        state["initial_points"] = np.asarray(src_pcd.points).copy()
        state["initial_normals"] = None
        if src_pcd.has_normals():
            src_pcd.normals = o3d.utility.Vector3dVector()
        if dst_pcd.has_normals():
            dst_pcd.normals = o3d.utility.Vector3dVector()
        ensure_normals_for_method()
        vis.update_geometry(dst_pcd)
        vis.update_geometry(src_pcd)
        vis.poll_events()
        vis.update_renderer()
        update_title(vis)
        print(f"loaded point sets from {args.io_path}")
        return False

    def set_method(method):
        def _callback(vis):
            state["method"] = method
            ensure_normals_for_method()
            update_title(vis)
            return False
        return _callback

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="ICP Demo", width=1280, height=720)
    vis.add_geometry(dst_pcd)
    vis.add_geometry(src_pcd)
    opt = vis.get_render_option()
    opt.point_size = max(1.0, args.point_size)
    opt.background_color = np.asarray([0.97, 0.97, 0.97])

    vis.register_key_callback(ord("s"), run_icp)
    vis.register_key_callback(ord("S"), run_icp)
    vis.register_key_callback(ord("r"), reset_view)
    vis.register_key_callback(ord("R"), reset_view)
    vis.register_key_callback(ord("1"), set_method("point2point"))
    vis.register_key_callback(ord("2"), set_method("point2plane"))
    vis.register_key_callback(ord("3"), set_method("plane2plane"))
    vis.register_key_callback(ord("p"), save_points)
    vis.register_key_callback(ord("P"), save_points)
    vis.register_key_callback(ord("l"), load_points)
    vis.register_key_callback(ord("L"), load_points)

    update_title(vis)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    main()
