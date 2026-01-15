#!/usr/bin/env python3
"""
Practice 08: Pose Graph Optimization with PyPose
- Generate 200 camera poses along circular trajectory
- Introduce drift from 10th camera onwards
- Visualize with Open3D
- Press 'F' to perform pose graph optimization using PyPose (loop closure)
"""

import numpy as np
import open3d as o3d
import torch
import pypose as pp
from pypose.optim import LM
import copy


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion [w, x, y, z]"""
    from scipy.spatial.transform import Rotation
    rot = Rotation.from_matrix(R)
    quat = rot.as_quat()  # Returns [x, y, z, w]
    return np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z]


def quaternion_to_rotation_matrix(q):
    """Convert quaternion [w, x, y, z] to rotation matrix"""
    from scipy.spatial.transform import Rotation
    quat_scipy = [q[1], q[2], q[3], q[0]]  # Convert [w,x,y,z] to [x,y,z,w]
    return Rotation.from_quat(quat_scipy).as_matrix()


def generate_circular_trajectory(num_poses=200, radius=5.0, height=0.0):
    """
    Generate camera poses along a circular trajectory
    Returns: list of (position, rotation_matrix) tuples
    """
    poses = []
    
    for i in range(num_poses):
        # Angle along the circle
        angle = 2 * np.pi * i / num_poses
        
        # Position on circle
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        z = height
        position = np.array([x, y, z])
        
        # Camera looks towards center of circle (0, 0, 0)
        # Forward direction from camera to center
        center = np.array([0, 0, 0])
        forward = center - position
        forward = forward / np.linalg.norm(forward)
        
        # Up direction (world Z-axis)
        world_up = np.array([0, 0, 1])
        
        # Right direction (cross product: world_up x forward for right-handed system)
        right = np.cross(world_up, forward)
        if np.linalg.norm(right) < 1e-6:
            # Handle degenerate case when forward is parallel to world_up
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)
        
        # Recompute up to ensure orthogonality (forward x right for right-handed system)
        up = np.cross(forward, right)
        up = up / np.linalg.norm(up)
        
        # Construct rotation matrix as columns [right, up, forward]
        # This ensures right-handed coordinate system: right × up = forward
        rotation = np.column_stack([right, up, forward])
        
        # Verify it's a valid rotation matrix
        det = np.linalg.det(rotation)
        if det < 0.99:  # Should be very close to 1.0
            print(f"Warning: Invalid rotation at pose {i}, det={det}")
        
        poses.append((position, rotation))
    
    return poses


def add_drift_to_poses(poses, drift_start_idx=10, drift_trans_per_step=0.01, drift_rot_per_step=0.002):
    """
    Add cumulative drift to poses starting from drift_start_idx
    
    Args:
        poses: list of (position, rotation) tuples
        drift_start_idx: index to start adding drift
        drift_trans_per_step: amount of translational drift per step (in meters)
        drift_rot_per_step: amount of rotational drift per step (in radians)
    
    Returns: list of drifted poses
    """
    from scipy.spatial.transform import Rotation
    drifted_poses = []
    cumulative_drift_rot = np.eye(3)
    
    for i, (pos, rot) in enumerate(poses):
        if i >= drift_start_idx:
            # Add translational drift (outward from the origin in XY plane).
            # Use a linearly increasing magnitude so drift_trans_per_step is easy
            # to interpret: at step k, drift magnitude = k * drift_trans_per_step.
            k = (i - drift_start_idx + 1)
            xy_norm = np.linalg.norm(pos[:2])
            if xy_norm > 1e-6:
                drift_direction = np.array([pos[0] / xy_norm, pos[1] / xy_norm, 0.0])
            else:
                drift_direction = np.array([1.0, 0.0, 0.0])
            drift_offset = drift_direction * (k * drift_trans_per_step)
            
            # Add rotational drift (yaw rotation) - much smaller
            drift_angle = drift_rot_per_step  # radians
            drift_rot = Rotation.from_euler('z', drift_angle).as_matrix()
            cumulative_drift_rot = drift_rot @ cumulative_drift_rot
            
            # Apply cumulative drift
            new_pos = pos + drift_offset
            new_rot = cumulative_drift_rot @ rot
        else:
            new_pos = pos
            new_rot = rot
        
        drifted_poses.append((new_pos, new_rot))
    
    return drifted_poses


def pose_to_transform(position, rotation):
    """Convert position and rotation to 4x4 transformation matrix"""
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = position
    return transform


def transform_to_pose(transform):
    """Convert 4x4 transformation matrix to position and rotation"""
    position = transform[:3, 3]
    rotation = transform[:3, :3]
    return position, rotation


def create_camera_frustum(scale=0.2):
    """Create a camera frustum mesh for visualization"""
    # Define frustum vertices (camera coordinate system)
    vertices = np.array([
        [0, 0, 0],           # Camera center
        [-1, -1, 2],         # Bottom-left
        [1, -1, 2],          # Bottom-right
        [1, 1, 2],           # Top-right
        [-1, 1, 2],          # Top-left
    ]) * scale
    
    # Define lines connecting vertices
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # From center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Image plane rectangle
    ]
    
    # Create line set
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    
    return line_set


def create_trajectory_line(poses, color=[1, 0, 0]):
    """Create a line connecting camera positions"""
    positions = np.array([pos for pos, _ in poses])
    
    # Create lines connecting consecutive poses
    lines = [[i, i+1] for i in range(len(positions)-1)]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(positions)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))
    
    return line_set


def optimize_pose_graph_pypose(
    poses,
    loop_closure_idx=(0, 199),
    num_iterations=500,
    w_odom=0.1,  # Lower weight - we care more about loop closure
    w_loop_trans=1000.0,  # Strong loop closure constraint
    w_loop_rot=1000.0,  # Strong loop closure constraint
    loop_tol_trans=1e-3,
    loop_tol_rot=1e-3,
    fixed_anchor_idx=29,  # Fix pose 0 and pose 29 (30th camera)
):
    """
    Perform pose graph optimization using PyPose
    
    Setup:
    - Fix pose 0 (first camera) and pose 29 (30th camera) as anchors
    - Single loop closure: pose 0 (first) should equal pose N-1 (last)
    - This closes the circular trajectory
    
    Args:
        poses: list of (position, rotation_matrix) tuples
        loop_closure_idx: tuple of (start_idx, end_idx) for loop closure
        num_iterations: number of optimization iterations
        fixed_anchor_idx: second fixed pose index (default 29 = 30th camera)
    
    Returns: generator yielding optimized poses at each iteration
    """
    n_poses = len(poses)
    
    # Convert poses to PyPose SE3 format
    pose_tensors = []
    for pos, rot in poses:
        quat = rotation_matrix_to_quaternion(rot)  # [w, x, y, z]
        se3_vec = np.concatenate([pos, quat])
        pose_tensors.append(se3_vec)
    
    pose_array = torch.tensor(np.array(pose_tensors), dtype=torch.float32)
    pose_array_reordered = torch.cat([
        pose_array[:, :3],  # translation
        pose_array[:, 4:7],  # qx, qy, qz
        pose_array[:, 3:4]   # qw
    ], dim=1)
    
    # FIX pose 0 and pose 29 (30th camera) as anchors
    # Store all poses, but only optimize non-fixed ones
    fixed_pose_0 = pp.SE3(pose_array_reordered[0:1].clone())
    fixed_pose_anchor = pp.SE3(pose_array_reordered[fixed_anchor_idx:fixed_anchor_idx+1].clone())
    
    # Create list of optimizable pose indices (exclude 0 and fixed_anchor_idx)
    optimizable_indices = [i for i in range(n_poses) if i != 0 and i != fixed_anchor_idx]
    optimizable_data = pose_array_reordered[optimizable_indices]
    optimizable_poses = pp.Parameter(pp.SE3(optimizable_data.clone()))
    
    # Higher learning rate for faster convergence
    optimizer = torch.optim.Adam([optimizable_poses], lr=5e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.7, patience=30
    )
    
    # Single loop closure: pose 0 (first) == pose N-1 (last)
    start_idx, end_idx = loop_closure_idx
    loop_start = start_idx  # 0 (fixed)
    loop_end = end_idx  # N-1 (last pose)
    
    best_loss = float('inf')
    stall_count = 0
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Helper to get pose (accounting for fixed poses)
        def get_pose(idx):
            if idx == 0:
                return fixed_pose_0[0]
            elif idx == fixed_anchor_idx:
                return fixed_pose_anchor[0]
            else:
                # Find position in optimizable_poses
                opt_idx = optimizable_indices.index(idx)
                return optimizable_poses[opt_idx]
        
        # Odometry constraints - relative motion between consecutive poses
        odometry_loss = 0
        for i in range(n_poses - 1):
            T_i_orig = pp.SE3(pose_array_reordered[i:i+1])
            T_j_orig = pp.SE3(pose_array_reordered[i+1:i+2])
            rel_orig = T_i_orig.Inv() @ T_j_orig
            
            rel_current = get_pose(i).Inv() @ get_pose(i + 1)
            
            diff = rel_current @ rel_orig.Inv()
            trans_error = diff.translation().norm() ** 2
            rot_error = diff.rotation().Log().norm() ** 2
            odometry_loss += trans_error + rot_error
        
        odometry_loss = odometry_loss / (n_poses - 1)
        
        # SINGLE LOOP CLOSURE: pose[0] should equal pose[N-1]
        pose_first = get_pose(loop_start)
        pose_last = get_pose(loop_end)
        diff_loop = pose_last @ pose_first.Inv()
        loop_trans_error = diff_loop.translation().norm() ** 2
        loop_rot_error = diff_loop.rotation().Log().norm() ** 2
        
        loop_closure_loss = (
            w_loop_trans * loop_trans_error +
            w_loop_rot * loop_rot_error
        )
        
        # Total loss - loop closure dominates
        total_loss = w_odom * odometry_loss + loop_closure_loss
        
        # Optimize
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([optimizable_poses], max_norm=2.0)
        optimizer.step()
        
        scheduler.step(total_loss.item())
        
        # Check convergence
        with torch.no_grad():
            loop_trans = torch.sqrt(loop_trans_error).item()
            loop_rot = torch.sqrt(loop_rot_error).item()
            
            loop_closed = (
                loop_trans < loop_tol_trans and loop_rot < loop_tol_rot
            )
            
            # Build current poses (including fixed poses)
            current_poses = []
            for i in range(n_poses):
                se3_pose = get_pose(i)
                translation = se3_pose.translation().numpy()
                rotation_quat = se3_pose.rotation().numpy()
                quat_wxyz = np.array([rotation_quat[3], rotation_quat[0], 
                                     rotation_quat[1], rotation_quat[2]])
                rotation_matrix = quaternion_to_rotation_matrix(quat_wxyz)
                current_poses.append((translation, rotation_matrix))
            
            yield current_poses, total_loss.item()
            
            # Early stopping if converged
            if loop_closed:
                print(f"{'='*60}")
                print(f"CONVERGED at iteration {iteration + 1}/{num_iterations}")
                print(f"Loop closure (0↔{loop_end}): trans={loop_trans:.6f}m, rot={loop_rot:.6f}rad")
                print(f"Final loss: {total_loss.item():.6f}")
                print(f"{'='*60}")
                break
            
            # Check for stalling
            if total_loss.item() < best_loss - 1e-5:
                best_loss = total_loss.item()
                stall_count = 0
            else:
                stall_count += 1
                if stall_count > 100:
                    print(f"Optimization stalled at iteration {iteration + 1}")
                    break
    
    # Final report
    with torch.no_grad():
        print(f"{'='*60}")
        print(f"Optimization completed: {iteration + 1} iterations")
        print(f"Fixed anchors: pose 0 and pose {fixed_anchor_idx}")
        print(f"Loop closure (0↔{loop_end}): trans={loop_trans:.6f}m, rot={loop_rot:.6f}rad")
        print(f"Final loss: {total_loss.item():.6f}")
        if not loop_closed:
            print("WARNING: Did not fully converge to tolerance!")
        print(f"{'='*60}")



class PoseGraphVisualizer:
    """Interactive visualizer for pose graph optimization"""
    
    def __init__(self, original_poses, drifted_poses, drift_start_idx=None):
        self.original_poses = original_poses
        self.drifted_poses = drifted_poses
        self.drifted_poses_original = copy.deepcopy(drifted_poses)  # Keep original for reference
        self.optimized_poses = None
        self.show_optimized = False
        self.optimization_running = False
        self.optimization_paused = False
        self.single_step = False
        self.current_iteration = 0
        self.total_iterations = 500
        self.optimization_generator = None
        self.drift_start_idx = drift_start_idx
        
        # Create visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name='Pose Graph Optimization', width=1280, height=720)
        
        # Set black background
        opt = self.vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])
        
        # Register key callbacks
        self.vis.register_key_callback(ord("F"), self.optimize_callback)
        self.vis.register_key_callback(ord("P"), self.pause_callback)
        self.vis.register_key_callback(ord("S"), self.step_callback)
        self.vis.register_key_callback(ord("R"), self.reset_callback)
        self.vis.register_key_callback(ord("H"), self.print_help)
        
        # Create geometries
        self.geometries_initialized = False
        
    def initialize_view(self):
        """Set up initial view after geometries are added"""
        ctr = self.vis.get_view_control()
        # Set camera looking at the scene from above and outside
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_front([8, 8, 5])  # View from outside the circle
        ctr.set_zoom(0.3)
    
    def create_camera_geometries(self, poses, color, every_n=5):
        """Create camera frustums and trajectory for given poses"""
        geometries = []
        
        # Trajectory line
        traj_line = create_trajectory_line(poses, color=color)
        geometries.append(('trajectory', traj_line))
        
        # Camera frustums (show every N-th camera)
        for i in range(0, len(poses), every_n):
            pos, rot = poses[i]
            frustum = create_camera_frustum(scale=0.15)
            frustum.paint_uniform_color(color)
            
            # Transform frustum to camera pose
            transform = pose_to_transform(pos, rot)
            frustum.transform(transform)
            
            geometries.append((f'camera_{i}', frustum))
        
        # Highlight fixed anchors: pose 0 (first) and pose 29 (30th)
        # Also highlight last pose to show loop closure
        for i in [0, 29, len(poses)-1]:
            pos, rot = poses[i]
            frustum = create_camera_frustum(scale=0.25)
            if i == 0 or i == len(poses)-1:
                frustum.paint_uniform_color([1, 1, 0])  # Yellow for first/last
            else:
                frustum.paint_uniform_color([1, 0, 1])  # Magenta for 30th camera
            transform = pose_to_transform(pos, rot)
            frustum.transform(transform)
            geometries.append((f'highlight_{i}', frustum))
        
        return geometries
    
    def update_geometries(self, quiet=False):
        """Update visualization geometries"""
        self.vis.clear_geometries()
        
        # Always show drifted trajectory in red (reference)
        drifted_geoms = self.create_camera_geometries(
            self.drifted_poses,
            color=[1, 0, 0],  # Red
            every_n=10  # Show fewer cameras to reduce clutter
        )
        
        # Add drifted geometries
        for name, geom in drifted_geoms:
            self.vis.add_geometry(geom)
        
        # If optimization has started, overlay optimized trajectory in green
        if self.optimized_poses is not None:
            optimized_geoms = self.create_camera_geometries(
                self.optimized_poses, 
                color=[0, 1, 0],  # Green
                every_n=10
            )
            for name, geom in optimized_geoms:
                self.vis.add_geometry(geom)
            
            # Add single loop closure constraint line
            # Line from pose 0 (first) to pose N-1 (last)
            loop_line = self.create_loop_closure_line(
                self.optimized_poses[0][0], 
                self.optimized_poses[-1][0],
                [1, 1, 0]  # Yellow
            )
            self.vis.add_geometry(loop_line)
            
            if not quiet:
                print("Displaying: Red=Drifted, Green=Optimized")
        else:
            if not quiet:
                print("Displaying: Red=Drifted (press F to optimize)")
        
        # Add coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
        self.vis.add_geometry(coord_frame)
        
        # Update renderer
        self.vis.poll_events()
        self.vis.update_renderer()
    
    def create_loop_closure_line(self, pos1, pos2, color):
        """Create a line showing loop closure constraint"""
        points = np.array([pos1, pos2])
        lines = [[0, 1]]
        
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector([color])
        
        return line_set
    
    def optimize_callback(self, vis):
        """Callback for 'F' key: start iterative optimization"""
        if not self.optimization_running and self.current_iteration == 0:
            print("\n[F] pressed: Starting PyPose pose graph optimization...")
            print(f"Optimization will run for {self.total_iterations} iterations")
            
            # Calculate initial error
            initial_error = np.linalg.norm(self.drifted_poses[-1][0] - self.drifted_poses[0][0])
            print(f"Initial loop closure error: {initial_error:.3f}")
            
            # Create optimization generator with balanced default parameters
            self.optimization_generator = optimize_pose_graph_pypose(
                self.drifted_poses_original, 
                num_iterations=self.total_iterations,
                # Use default balanced weights for proper convergence:
                # w_odom=1.0, w_loop_trans=100.0, w_loop_rot=100.0
                # loop_tol_trans=1e-3, loop_tol_rot=1e-3
            )
            
            self.optimization_running = True
            self.current_iteration = 0
            self.show_optimized = True
        elif self.optimization_running:
            print("\n[F] pressed: Optimization already running, ignoring")
        elif self.current_iteration >= self.total_iterations:
            print("\n[F] pressed: Optimization already complete, ignoring")
        return False
    
    def pause_callback(self, vis):
        """Callback for 'P' key: pause/resume optimization"""
        if self.optimization_running:
            self.optimization_paused = not self.optimization_paused
            status = "PAUSED" if self.optimization_paused else "RESUMED"
            print(f"\n[P] pressed: Optimization {status}")
        else:
            print("\n[P] pressed: No optimization running")
        return False
    
    def step_callback(self, vis):
        """Callback for 'S' key: single step optimization"""
        if self.optimization_running:
            if not self.optimization_paused:
                self.optimization_paused = True
                print("\n[S] pressed: Entering single-step mode (paused)")
            self.single_step = True
            print(f"  Stepping to iteration {self.current_iteration + 1}")
        else:
            print("\n[S] pressed: No optimization running. Press 'F' first.")
        return False
    
    def reset_callback(self, vis):
        """Callback for 'R' key: reset optimization"""
        if self.optimization_running or self.optimized_poses is not None:
            print("\n[R] pressed: Resetting optimization...")
            self.optimization_running = False
            self.optimization_paused = False
            self.single_step = False
            self.current_iteration = 0
            self.optimized_poses = None
            self.optimization_generator = None
            self.show_optimized = False
            self.update_geometries()
            print("  Ready to optimize again (press F)")
        else:
            print("\n[R] pressed: Nothing to reset")
        return False
    
    def optimization_step(self):
        """Perform one iteration of optimization and update display"""
        if not self.optimization_running:
            return False
        
        # Check if paused (and not single-stepping)
        if self.optimization_paused and not self.single_step:
            return True  # Keep running but don't advance
        
        # Reset single step flag after using it
        if self.single_step:
            self.single_step = False
            
        if self.current_iteration >= self.total_iterations:
            self.optimization_running = False
            return False
        
        try:
            # Get next optimization result from generator
            self.optimized_poses, loss = next(self.optimization_generator)
            
            # Update visualization (quiet mode to reduce print spam)
            self.update_geometries(quiet=True)
            
            # Progress update - show single loop closure error
            loop_error = np.linalg.norm(self.optimized_poses[-1][0] - self.optimized_poses[0][0])
            
            status = " [PAUSED]" if self.optimization_paused else ""
            print(
                f"Iter {self.current_iteration + 1:3d}/{self.total_iterations}: "
                f"Loop(0↔199)={loop_error:.4f}, Loss={loss:.6f}{status}"
            )
            
            self.current_iteration += 1
            
            # Check if optimization is complete
            if self.current_iteration >= self.total_iterations:
                self.optimization_running = False
                final_error = np.linalg.norm(self.optimized_poses[-1][0] - self.optimized_poses[0][0])
                initial_error = np.linalg.norm(self.drifted_poses[-1][0] - self.drifted_poses[0][0])
                print(f"\n✓ PyPose optimization complete!")
                print(f"  Final loop closure error: {final_error:.4f}")
                print(f"  Improvement: {initial_error - final_error:.4f}")
                print(f"  Final loss: {loss:.6f}")
                return False
            
            return True
        except StopIteration:
            # Generator stopped early (converged)
            self.optimization_running = False
            if self.optimized_poses is not None:
                final_error = np.linalg.norm(self.optimized_poses[-1][0] - self.optimized_poses[0][0])
                initial_error = np.linalg.norm(self.drifted_poses[-1][0] - self.drifted_poses[0][0])
                print(f"\n✓ Optimization converged early!")
                print(f"  Final loop closure error: {final_error:.4f}")
                print(f"  Improvement: {initial_error - final_error:.4f}")
            return False
    
    def print_help(self, vis):
        """Print help information"""
        print("\n" + "="*60)
        print("=== Keyboard Controls ===")
        print("="*60)
        print("  F: Start pose graph optimization")
        print("  P: Pause/Resume optimization")
        print("  S: Single step (when paused)")
        print("  R: Reset optimization")
        print("  H: Print this help")
        print("  Q/ESC: Quit")
        print("\nMouse Controls:")
        print("  Left-drag: Rotate view")
        print("  Right-drag/Scroll: Zoom")
        print("  Middle-drag: Pan view")
        print("\nVisualization:")
        print("  Red trajectory: Drifted poses (with accumulated error)")
        print("  Green trajectory: Optimized poses (being corrected)")
        print("  Yellow line: Loop closure constraint (first ↔ last)")
        print("  Yellow cameras: First and last poses (should match)")
        print("  Magenta camera: 30th pose (fixed anchor)")
        print("="*60)
        return False
    
    def run(self):
        """Run the interactive visualization"""
        print("\n" + "="*60)
        print("=== Pose Graph Optimization - Step-by-Step Visualization ===")
        print("="*60)
        print(f"\nGenerated {len(self.drifted_poses)} camera poses in circular trajectory")
        if self.drift_start_idx is not None:
            print(f"Drift introduced from camera {self.drift_start_idx} onwards")
        drift_error = np.linalg.norm(self.drifted_poses[-1][0] - self.drifted_poses[0][0])
        print(f"Initial loop closure error: {drift_error:.3f} meters")
        print("\n" + "="*60)
        print("Quick Start:")
        print("  1. Press 'F' to start optimization")
        print("  2. Press 'P' to pause and watch step-by-step")
        print("  3. Press 'S' to advance one step at a time")
        print("  4. Press 'H' for full help")
        print("="*60)
        
        # Add initial geometries
        self.update_geometries()
        
        # Set initial view after geometries are added
        self.initialize_view()
        
        # Custom animation loop to handle iterative optimization
        def animation_callback(vis):
            # Perform optimization step if running
            if self.optimization_running:
                self.optimization_step()
            # Return False to keep rendering, True would stop
            return False
        
        # Register animation callback
        self.vis.register_animation_callback(animation_callback)
        
        self.vis.run()
        self.vis.destroy_window()


def main():
    print("=== Practice 08: Pose Graph Optimization with PyPose ===\n")
    
    # Generate ground truth circular trajectory
    print("Generating circular trajectory...")
    num_poses = 200
    radius = 7.0
    original_poses = generate_circular_trajectory(num_poses, radius=radius)
    
    print(f"Generated {len(original_poses)} poses along circle (radius={radius})")
    
    # Add drift starting from pose 10
    print("\nAdding drift from pose 10 onwards...")
    drift_start = 10
    drift_trans = 0.01  # Translation drift per step (meters) - 10x increased
    drift_rot = 0.001   # Rotation drift per step (radians)
    drifted_poses = add_drift_to_poses(original_poses, drift_start_idx=drift_start, 
                                      drift_trans_per_step=drift_trans, 
                                      drift_rot_per_step=drift_rot)
    
    # Calculate drift statistics
    drift_at_end = np.linalg.norm(drifted_poses[-1][0] - original_poses[-1][0])
    print(f"Total drift at final pose: {drift_at_end:.3f} units")
    
    # Visualize
    visualizer = PoseGraphVisualizer(original_poses, drifted_poses, drift_start_idx=drift_start)
    visualizer.run()


if __name__ == "__main__":
    main()
