#!/usr/bin/env python3
"""
Practice 07: Structure from Motion (SfM) using pycolmap
- Load image set from resource folder
- Perform incremental reconstruction
- Visualize reconstructed point cloud and camera poses in 3D
"""

import os
import cv2
import numpy as np
import pycolmap
from pathlib import Path
import open3d as o3d
import json
import subprocess
import shutil
import argparse


def load_images_from_folder(folder_path):
    """
    Load all images from specified folder
    Returns: list of (image_path, image) tuples
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    images = []
    
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder {folder_path} does not exist")
        return images
    
    for img_path in sorted(folder.glob('*')):
        # Skip hidden files and non-image files
        if img_path.name.startswith('.'):
            continue
        if img_path.suffix.lower() in image_extensions:
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append((str(img_path), img))
                print(f"Loaded: {img_path.name} - {img.shape}")
    
    print(f"\nTotal images loaded: {len(images)}")
    return images


def run_sfm_reconstruction(image_folder, output_path, dense=False, dense_method='stereo', generate_mesh=False, mesh_method='poisson', force_recompute=False):
    """
    Run Structure from Motion reconstruction using pycolmap
    Args:
        image_folder: Path to input images
        output_path: Path to output directory
        dense: If True, perform dense reconstruction (slower but more points)
        dense_method: 'colmap' (requires CUDA), 'stereo' (CPU-based), or 'openmvs'
        generate_mesh: If True, generate mesh from point cloud
        mesh_method: 'poisson' (smooth) or 'ball_pivoting' (preserves details)
        force_recompute: If True, delete existing data and recompute from scratch
    Returns: reconstruction object
    """
    print("\n=== Starting SfM Reconstruction ===")
    print(f"Mode: {'Dense' if dense else 'Sparse'} reconstruction")
    if dense:
        print(f"Dense method: {dense_method}")
    if generate_mesh:
        print(f"Mesh generation: {mesh_method}")
    
    # Create output directory
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    database_path = output_path / "database.db"
    
    # Check if we should clean existing data
    if force_recompute and database_path.exists():
        print("\n⚠️  Force recompute enabled - cleaning existing data...")
        # Remove database
        database_path.unlink()
        print(f"  ✓ Removed {database_path}")
        
        # Remove reconstruction folders
        for subdir in output_path.glob('[0-9]*'):
            if subdir.is_dir():
                shutil.rmtree(subdir)
                print(f"  ✓ Removed {subdir}")
        
        # Remove dense/openmvs folders
        for folder_name in ['dense', 'openmvs']:
            folder = output_path / folder_name
            if folder.exists():
                shutil.rmtree(folder)
                print(f"  ✓ Removed {folder}")
    
    # Warn if database exists
    elif database_path.exists():
        print(f"\n⚠️  Existing database found: {database_path}")
        print("  COLMAP may reuse cached features if images haven't changed.")
        print("  Set FORCE_RECOMPUTE=True to force clean reconstruction.")
    
    # Step 1: Feature extraction
    print("\n[1/4] Extracting features...")
    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_folder),
        camera_mode=pycolmap.CameraMode.AUTO
    )
    
    # Step 2: Feature matching
    print("[2/4] Matching features...")
    pycolmap.match_exhaustive(
        database_path=str(database_path)
    )
    
    # Step 3: Incremental mapper (sparse reconstruction)
    print("[3/4] Running incremental reconstruction...")
    maps = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_folder),
        output_path=str(output_path)
    )
    
    print(f"[4/4] Reconstruction complete! Generated {len(maps)} models")
    
    # Get the best reconstruction (usually the first one)
    if len(maps) == 0:
        print("Warning: No reconstruction generated")
        return None
    
    reconstruction = maps[0]
    
    # Step 4: Dense reconstruction (if requested)
    if dense:
        if dense_method == 'colmap':
            # COLMAP's GPU-based dense reconstruction
            print("\n=== Starting Dense Reconstruction (COLMAP) ===")
            sparse_model_path = output_path / "0"
            dense_workspace = output_path / "dense"
            dense_workspace.mkdir(exist_ok=True)
            
            print("[1/2] Computing stereo depth maps...")
            try:
                pycolmap.patch_match_stereo(
                    workspace_path=str(dense_workspace),
                    workspace_format="COLMAP",
                    pmvs_option_name="option-all"
                )
                
                print("[2/2] Fusing depth maps into dense point cloud...")
                pycolmap.stereo_fusion(
                    output_path=str(dense_workspace / "fused.ply"),
                    workspace_path=str(dense_workspace),
                    workspace_format="COLMAP",
                    pmvs_option_name="option-all",
                    input_type="geometric"
                )
                
                # Load dense point cloud
                dense_pcd = o3d.io.read_point_cloud(str(dense_workspace / "fused.ply"))
                if len(dense_pcd.points) > 0:
                    print(f"Dense reconstruction: {len(dense_pcd.points)} points generated")
                    # Store dense point cloud in reconstruction object for later use
                    reconstruction.dense_points = np.asarray(dense_pcd.points)
                    reconstruction.dense_colors = np.asarray(dense_pcd.colors)
                else:
                    print("Warning: Dense reconstruction generated no points")
                    
            except Exception as e:
                print(f"Dense reconstruction failed: {e}")
                print("Falling back to sparse reconstruction only")
        
        elif dense_method == 'stereo':
            # CPU-based stereo matching
            try:
                dense_points, dense_colors = dense_reconstruction_stereo(
                    reconstruction, image_folder, output_path
                )
                
                if dense_points is not None:
                    reconstruction.dense_points = dense_points
                    reconstruction.dense_colors = dense_colors
                else:
                    print("Falling back to sparse reconstruction only")
                    
            except Exception as e:
                print(f"Dense reconstruction failed: {e}")
                print("Falling back to sparse reconstruction only")
        
        elif dense_method == 'openmvs':
            # OpenMVS-based dense reconstruction
            try:
                dense_points, dense_colors = dense_reconstruction_openmvs(
                    reconstruction, image_folder, output_path
                )
                
                if dense_points is not None:
                    reconstruction.dense_points = dense_points
                    reconstruction.dense_colors = dense_colors
                else:
                    print("Falling back to sparse reconstruction only")
                    
            except Exception as e:
                print(f"Dense reconstruction failed: {e}")
                print("Falling back to sparse reconstruction only")
    
    # Step 5: Mesh generation (if requested)
    mesh = None
    openmvs_mesh_path = None
    
    if generate_mesh:
        # Determine which points to use for meshing
        if hasattr(reconstruction, 'dense_points') and reconstruction.dense_points is not None:
            mesh_points = reconstruction.dense_points
            mesh_colors = reconstruction.dense_colors
            print(f"\nGenerating mesh from {len(mesh_points)} dense points...")
        else:
            # Use sparse points
            mesh_points, mesh_colors = reconstruction_to_arrays(reconstruction)
            print(f"\nGenerating mesh from {len(mesh_points)} sparse points...")
        
        # Generate mesh using Open3D
        mesh_path = output_path / "0" / f"mesh_{mesh_method}.ply"
        mesh = generate_mesh_from_point_cloud(
            mesh_points, mesh_colors, 
            method=mesh_method,
            output_path=mesh_path
        )
        
        # If using OpenMVS, also generate OpenMVS mesh
        if mesh is not None and dense_method == 'openmvs' and hasattr(reconstruction, 'dense_points'):
            mvs_folder = output_path / "openmvs"
            if mvs_folder.exists():
                openmvs_mesh_path = generate_mesh_openmvs(mvs_folder, refine=True, texture=True)
    
    return reconstruction, mesh, openmvs_mesh_path


def reconstruction_to_arrays(reconstruction):
    """
    Extract points and colors from pycolmap reconstruction
    Returns: points (Nx3), colors (Nx3)
    """
    points = []
    colors = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        # Normalize RGB color to [0, 1]
        colors.append(point3D.color / 255.0)
    
    return np.array(points), np.array(colors)


def dense_reconstruction_stereo(reconstruction, image_folder, output_path, 
                                max_pairs=10, num_disparities=128, block_size=11):
    """
    CPU-based dense reconstruction using OpenCV stereo matching
    Alternative to COLMAP's GPU-based patch match stereo
    
    Args:
        reconstruction: COLMAP reconstruction with camera poses
        image_folder: Path to images
        output_path: Output directory
        max_pairs: Maximum number of image pairs to process
        num_disparities: Number of disparity levels (must be divisible by 16)
        block_size: Size of the block for matching (odd number, typically 5-21)
    
    Returns: Dense point cloud (Nx3), colors (Nx3)
    """
    print("\n=== CPU-Based Dense Reconstruction (Stereo Matching) ===")
    
    image_folder = Path(image_folder)
    output_path = Path(output_path)
    
    # Get list of registered images sorted by position
    images_list = []
    for image_id, image in reconstruction.images.items():
        cam_center = -image.cam_from_world().rotation.matrix().T @ image.cam_from_world().translation
        images_list.append((image_id, image, cam_center))
    
    print(f"Processing {len(images_list)} registered images")
    
    # Create stereo matcher
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    
    all_points = []
    all_colors = []
    
    # Process image pairs
    pairs_processed = 0
    for i in range(len(images_list) - 1):
        if pairs_processed >= max_pairs:
            break
        
        # Get consecutive image pair
        img1_id, img1, center1 = images_list[i]
        img2_id, img2, center2 = images_list[i + 1]
        
        # Check baseline (distance between cameras)
        baseline = np.linalg.norm(center1 - center2)
        
        # Load images
        img1_path = image_folder / img1.name
        img2_path = image_folder / img2.name
        
        if not img1_path.exists() or not img2_path.exists():
            continue
        
        left_img = cv2.imread(str(img1_path))
        right_img = cv2.imread(str(img2_path))
        
        if left_img is None or right_img is None:
            continue
        
        # Resize for faster processing
        scale = 0.5
        left_img_small = cv2.resize(left_img, None, fx=scale, fy=scale)
        right_img_small = cv2.resize(right_img, None, fx=scale, fy=scale)
        
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_img_small, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img_small, cv2.COLOR_BGR2GRAY)
        
        print(f"  Pair {pairs_processed + 1}: {img1.name} <-> {img2.name} (baseline: {baseline:.3f})")
        
        # Compute disparity
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Get camera parameters
        camera1 = reconstruction.cameras[img1.camera_id]
        focal_length = camera1.focal_length * scale  # Scale focal length
        
        # Filter disparity (keep only valid values)
        valid_disparity = disparity > 0
        num_valid = np.sum(valid_disparity)
        
        if num_valid < 100:
            print(f"    Skipping: too few valid disparities ({num_valid})")
            continue
        
        # Compute depth from disparity (simple pinhole model)
        # For simplicity, use focal_length * baseline as rough depth scale
        depth_scale = focal_length * baseline
        depth = np.zeros_like(disparity)
        depth[valid_disparity] = depth_scale / (disparity[valid_disparity] + 1e-6)
        
        # Filter depth (remove unrealistic depths based on scene scale)
        scene_scale = np.median(np.linalg.norm(list(reconstruction.points3D.values())[0].xyz))
        min_depth = scene_scale * 0.1
        max_depth = scene_scale * 10.0
        valid_depth = (depth > min_depth) & (depth < max_depth) & valid_disparity
        
        num_valid_depth = np.sum(valid_depth)
        if num_valid_depth < 100:
            print(f"    Skipping: too few valid depths ({num_valid_depth})")
            continue
        
        # Get camera transformation
        camera_matrix = img1.cam_from_world()
        R = camera_matrix.rotation.matrix()
        t = camera_matrix.translation
        
        h, w = depth.shape
        camera_orig = reconstruction.cameras[img1.camera_id]
        cx = (camera_orig.principal_point_x if hasattr(camera_orig, 'principal_point_x') else camera_orig.width / 2) * scale
        cy = (camera_orig.principal_point_y if hasattr(camera_orig, 'principal_point_y') else camera_orig.height / 2) * scale
        
        # Create 3D points
        points_3d = []
        colors_3d = []
        
        step = 2  # Subsample for efficiency
        for v in range(0, h, step):
            for u in range(0, w, step):
                if valid_depth[v, u]:
                    # Camera coordinates
                    z = depth[v, u]
                    x = (u - cx) * z / focal_length
                    y = (v - cy) * z / focal_length
                    
                    # Transform to world coordinates
                    point_cam = np.array([x, y, z])
                    point_world = R.T @ (point_cam - t)
                    
                    points_3d.append(point_world)
                    colors_3d.append(left_img_small[v, u][::-1] / 255.0)  # BGR to RGB
        
        if len(points_3d) > 0:
            all_points.extend(points_3d)
            all_colors.extend(colors_3d)
            print(f"    Generated {len(points_3d)} points")
        
        pairs_processed += 1
    
    if len(all_points) == 0:
        print("Warning: No dense points generated")
        return None, None
    
    points = np.array(all_points)
    colors = np.array(all_colors)
    
    print(f"\n✓ Dense reconstruction complete: {len(points)} points from {pairs_processed} pairs")
    
    # Optional: filter outliers using statistical outlier removal
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print("  Filtering outliers...")
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    
    print(f"✓ After filtering: {len(points)} points")
    
    return points, colors


def dense_reconstruction_openmvs(reconstruction, image_folder, output_path):
    """
    Dense reconstruction using OpenMVS (if installed)
    OpenMVS must be installed separately: https://github.com/cdcseacave/openMVS
    
    Installation on macOS:
        brew install openmvs
    
    Or build from source following OpenMVS documentation
    
    Args:
        reconstruction: COLMAP reconstruction
        image_folder: Path to images
        output_path: Output directory
        
    Returns: Dense point cloud (Nx3), colors (Nx3) or None if failed
    """
    print("\n=== Dense Reconstruction with OpenMVS ===")
    
    output_path = Path(output_path)
    mvs_folder = output_path / "openmvs"
    mvs_folder.mkdir(exist_ok=True)
    
    # Check if OpenMVS is installed
    openmvs_commands = ['InterfaceCOLMAP', 'DensifyPointCloud', 'ReconstructMesh', 'RefineMesh', 'TextureMesh']
    openmvs_available = True
    
    for cmd in ['InterfaceCOLMAP', 'DensifyPointCloud']:
        if shutil.which(cmd) is None:
            print(f"  Warning: {cmd} not found in PATH")
            openmvs_available = False
    
    if not openmvs_available:
        print("\n  OpenMVS is not installed or not in PATH.")
        print("  To install OpenMVS:")
        print("    macOS: brew install openmvs")
        print("    Linux: Follow https://github.com/cdcseacave/openMVS/wiki/Building")
        print("    Windows: Download pre-built binaries")
        return None, None
    
    try:
        # Step 1: Convert COLMAP to OpenMVS format
        print("\n[1/3] Converting COLMAP reconstruction to OpenMVS format...")
        sparse_model = output_path / "0"
        mvs_scene = mvs_folder / "scene.mvs"
        
        cmd = [
            'InterfaceCOLMAP',
            '-i', str(sparse_model),
            '-o', str(mvs_scene),
            '--image-folder', str(image_folder)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            print(f"  Error converting to OpenMVS: {result.stderr}")
            return None, None
        
        print(f"  ✓ Converted to {mvs_scene}")
        
        # Step 2: Dense reconstruction
        print("\n[2/3] Running dense point cloud reconstruction...")
        dense_mvs = mvs_folder / "scene_dense.mvs"
        
        cmd = [
            'DensifyPointCloud',
            str(mvs_scene),
            '-o', str(dense_mvs),
            '--resolution-level', '1',  # 0=full res, 1=half res, 2=quarter res
            '--number-views', '4',      # Minimum number of views
            '--max-resolution', '3200'  # Maximum image resolution
        ]
        
        print("  This may take several minutes...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)  # 30 min timeout
        
        if result.returncode != 0:
            print(f"  Error during densification: {result.stderr}")
            return None, None
        
        # Step 3: Load dense point cloud
        print("\n[3/3] Loading dense point cloud...")
        dense_ply = mvs_folder / "scene_dense.ply"
        
        if not dense_ply.exists():
            print(f"  Warning: Dense PLY not found at {dense_ply}")
            return None, None
        
        pcd = o3d.io.read_point_cloud(str(dense_ply))
        
        if len(pcd.points) == 0:
            print("  Warning: Empty point cloud")
            return None, None
        
        print(f"  ✓ Loaded {len(pcd.points)} points")
        
        # Filter outliers
        print("  Filtering outliers...")
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        
        print(f"\n✓ OpenMVS dense reconstruction complete: {len(points)} points")
        
        return points, colors
        
    except subprocess.TimeoutExpired:
        print("  Error: OpenMVS process timed out")
        return None, None
    except Exception as e:
        print(f"  Error during OpenMVS reconstruction: {e}")
        return None, None


def reconstruction_to_arrays(reconstruction):
    """
    Extract points and colors from pycolmap reconstruction
    Returns: points (Nx3), colors (Nx3)
    """
    points = []
    colors = []
    
    for point3D_id, point3D in reconstruction.points3D.items():
        points.append(point3D.xyz)
        # Normalize RGB color to [0, 1]
        colors.append(point3D.color / 255.0)
    
    return np.array(points), np.array(colors)


def generate_mesh_from_point_cloud(points, colors, method='poisson', output_path=None):
    """
    Generate mesh from point cloud using Open3D
    
    Args:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
        method: 'poisson' (recommended) or 'ball_pivoting'
        output_path: Optional path to save mesh
        
    Returns: Open3D TriangleMesh or None if failed
    """
    print(f"\n=== Generating Mesh ({method}) ===")
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Estimate normals
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    pcd.orient_normals_consistent_tangent_plane(30)
    
    try:
        if method == 'poisson':
            print("Running Poisson surface reconstruction...")
            try:
                mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                    pcd, depth=9, width=0, scale=1.1, linear_fit=False
                )
                
                # Remove low density vertices (outliers)
                print("Filtering low-density vertices...")
                densities = np.asarray(densities)
                density_threshold = np.quantile(densities, 0.1)
                vertices_to_remove = densities < density_threshold
                mesh.remove_vertices_by_mask(vertices_to_remove)
            except Exception as poisson_error:
                print(f"Poisson reconstruction encountered errors, but may have produced partial results.")
                print(f"Consider using --mesh-method ball_pivoting for more robust meshing.")
                # Try to continue with whatever mesh was created
                if 'mesh' not in locals() or mesh is None or len(mesh.vertices) == 0:
                    raise poisson_error
            
        elif method == 'ball_pivoting':
            print("Running Ball Pivoting Algorithm...")
            # Estimate radius for ball pivoting
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radius = 1.5 * avg_dist
            radii = [radius, radius * 2, radius * 4]
            
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(radii)
            )
        else:
            print(f"Unknown method: {method}")
            return None
        
        # Clean up mesh
        print("Cleaning mesh...")
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        
        print(f"✓ Generated mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        
        # Save mesh if path provided
        if output_path:
            output_path = Path(output_path)
            o3d.io.write_triangle_mesh(str(output_path), mesh)
            print(f"✓ Saved mesh to {output_path}")
        
        return mesh
        
    except Exception as e:
        print(f"Error generating mesh: {e}")
        return None


def generate_mesh_openmvs(mvs_folder, refine=True, texture=True):
    """
    Generate mesh using OpenMVS (if available)
    Requires OpenMVS to be installed
    
    Args:
        mvs_folder: Path to OpenMVS workspace
        refine: Whether to refine the mesh
        texture: Whether to generate texture
        
    Returns: Path to mesh file or None if failed
    """
    print("\n=== Mesh Generation with OpenMVS ===")
    
    mvs_folder = Path(mvs_folder)
    dense_mvs = mvs_folder / "scene_dense.mvs"
    
    if not dense_mvs.exists():
        print(f"Error: Dense reconstruction not found at {dense_mvs}")
        return None
    
    # Check if ReconstructMesh is available
    if shutil.which('ReconstructMesh') is None:
        print("ReconstructMesh not found. Skipping OpenMVS mesh generation.")
        return None
    
    try:
        # Step 1: Reconstruct mesh
        print("\n[1/3] Reconstructing mesh...")
        mesh_mvs = mvs_folder / "scene_dense_mesh.mvs"
        
        cmd = [
            'ReconstructMesh',
            str(dense_mvs),
            '-o', str(mesh_mvs),
            '--smooth', '1'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        
        if result.returncode != 0:
            print(f"Error reconstructing mesh: {result.stderr}")
            return None
        
        mesh_ply = mvs_folder / "scene_dense_mesh.ply"
        print(f"  ✓ Mesh reconstructed: {mesh_ply}")
        
        # Step 2: Refine mesh (optional)
        if refine and shutil.which('RefineMesh') is not None:
            print("\n[2/3] Refining mesh...")
            refined_mvs = mvs_folder / "scene_dense_mesh_refine.mvs"
            
            cmd = [
                'RefineMesh',
                str(mesh_mvs),
                '-o', str(refined_mvs),
                '--scales', '1'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                mesh_mvs = refined_mvs
                mesh_ply = mvs_folder / "scene_dense_mesh_refine.ply"
                print(f"  ✓ Mesh refined: {mesh_ply}")
            else:
                print("  Warning: Mesh refinement failed, using unrefined mesh")
        
        # Step 3: Texture mesh (optional)
        if texture and shutil.which('TextureMesh') is not None:
            print("\n[3/3] Texturing mesh...")
            textured_mvs = mvs_folder / "scene_dense_mesh_texture.mvs"
            
            cmd = [
                'TextureMesh',
                str(mesh_mvs),
                '-o', str(textured_mvs),
                '--export-type', 'obj'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            
            if result.returncode == 0:
                textured_obj = mvs_folder / "scene_dense_mesh_texture.obj"
                print(f"  ✓ Mesh textured: {textured_obj}")
                return textured_obj
            else:
                print("  Warning: Mesh texturing failed")
        
        print(f"\n✓ OpenMVS mesh generation complete: {mesh_ply}")
        return mesh_ply
        
    except subprocess.TimeoutExpired:
        print("  Error: Mesh generation timed out")
        return None
    except Exception as e:
        print(f"  Error during mesh generation: {e}")
        return None


def get_camera_positions(reconstruction):
    """
    Extract camera positions and orientations
    Returns: camera_centers (Nx3), camera_directions (Nx3x3)
    """
    camera_centers = []
    camera_rotations = []
    
    for image_id, image in reconstruction.images.items():
        # Images in reconstruction.images are already registered
        # Get camera pose (world to camera transformation)
        cam_from_world = image.cam_from_world()
        R = cam_from_world.rotation.matrix()
        t = cam_from_world.translation
        
        # Convert to camera center in world coordinates
        camera_center = -R.T @ t
        camera_centers.append(camera_center)
        camera_rotations.append(R.T)  # Store as world orientation
    
    return np.array(camera_centers), camera_rotations


def save_point_cloud_ply(filepath, points, colors):
    """
    Save point cloud to PLY format
    """
    with open(filepath, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        # Write points
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = (colors[i] * 255).astype(int)
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    
    print(f"Saved {len(points)} points to PLY file")


def extract_and_save_descriptors(database_path, reconstruction, output_path):
    """
    Extract SIFT descriptors from COLMAP database and save with 2D-3D correspondences
    This allows practice07-track.py to use the original descriptors instead of re-extracting
    """
    print("\n=== Extracting SIFT Descriptors ===")
    
    # Open database using the correct API
    import sqlite3
    conn = sqlite3.connect(str(database_path))
    cursor = conn.cursor()
    
    all_descriptors = []
    descriptor_to_3d = {}
    image_info = {}
    descriptor_count = 0
    
    # For each registered image in reconstruction
    for image_id, image in reconstruction.images.items():
        image_name = image.name
        
        # Get descriptors from database directly via SQL
        cursor.execute("SELECT data FROM descriptors WHERE image_id = ?", (image_id,))
        result = cursor.fetchone()
        
        if result is None:
            print(f"  Warning: No descriptors for {image_name}")
            continue
        
        # Deserialize SIFT descriptors (stored as BLOB)
        # COLMAP stores descriptors as uint8 arrays of size Nx128
        descriptors_blob = result[0]
        descriptors = np.frombuffer(descriptors_blob, dtype=np.uint8).reshape(-1, 128).astype(np.float32)
        
        # Build mapping from point2D index to point3D
        point2d_to_3d = {}
        for point2D_idx in range(len(image.points2D)):
            point2D = image.points2D[point2D_idx]
            if point2D.point3D_id != -1 and point2D.point3D_id in reconstruction.points3D:
                point2d_to_3d[point2D_idx] = point2D.point3D_id
        
        # Extract descriptors with 3D correspondences
        image_descriptor_count = 0
        for point2D_idx, point3D_id in point2d_to_3d.items():
            if point2D_idx < len(descriptors):
                point3D = reconstruction.points3D[point3D_id]
                all_descriptors.append(descriptors[point2D_idx])
                descriptor_to_3d[descriptor_count] = point3D.xyz
                descriptor_count += 1
                image_descriptor_count += 1
        
        image_info[image_name] = {
            'total_keypoints': len(descriptors),
            'descriptors_with_3d': image_descriptor_count
        }
        
        print(f"  {image_name}: {len(descriptors)} keypoints, {image_descriptor_count} with 3D points")
    
    conn.close()
    
    if len(all_descriptors) == 0:
        print("Error: No descriptors with 3D correspondences found!")
        return False
    
    # Convert to numpy array
    all_descriptors = np.array(all_descriptors, dtype=np.float32)
    
    # Save descriptors and 3D points
    output_path = Path(output_path)
    descriptor_file = output_path / "sift_descriptors.npz"
    
    descriptor_indices = list(descriptor_to_3d.keys())
    points_3d = np.array([descriptor_to_3d[idx] for idx in descriptor_indices])
    
    np.savez(descriptor_file,
             descriptors=all_descriptors,
             descriptor_indices=np.array(descriptor_indices),
             points_3d=points_3d)
    
    # Save metadata
    metadata = {
        'num_images': len(image_info),
        'num_descriptors': len(all_descriptors),
        'num_unique_3d_points': len(set(tuple(pt) for pt in points_3d)),
        'images': image_info
    }
    
    metadata_file = output_path / "sift_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved {len(all_descriptors)} descriptors to {descriptor_file}")
    print(f"✓ Unique 3D points: {metadata['num_unique_3d_points']}")
    print(f"✓ Metadata saved to {metadata_file}")
    
    return True


def create_camera_frustum(center, rotation, scale=0.1, color=[1, 0, 0], image_path=None, load_texture=False):
    """
    Create a camera frustum mesh for visualization
    Args:
        center: Camera center in world coordinates
        rotation: Camera rotation matrix
        scale: Scale of the frustum
        color: Color of frustum lines
        image_path: Optional path to image to display as texture
        load_texture: If True, load and display image texture (may cause crashes on some systems)
    Returns: Tuple of (LineSet, Optional[TriangleMesh]) - frustum and optional textured plane
    """
    # Define frustum vertices in camera space
    # Camera looks down +Z axis
    w, h = scale * 0.5, scale * 0.4
    d = scale
    
    vertices = np.array([
        [0, 0, 0],           # 0: camera center
        [-w, -h, d],         # 1: bottom-left
        [w, -h, d],          # 2: bottom-right
        [w, h, d],           # 3: top-right
        [-w, h, d],          # 4: top-left
    ])
    
    # Transform vertices to world space
    vertices = (rotation @ vertices.T).T + center
    
    # Define edges of frustum
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
        [1, 2], [2, 3], [3, 4], [4, 1],  # Rectangle at image plane
    ]
    
    # Create LineSet
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector([color for _ in range(len(lines))])
    
    # Create textured plane if image is provided and texture loading is enabled
    image_plane = None
    if load_texture and image_path is not None and os.path.exists(image_path):
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is not None:
                # Convert BGR to RGB and normalize
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize image for performance (max 512 pixels on longest side)
                max_size = 512
                h_img, w_img = img.shape[:2]
                if max(h_img, w_img) > max_size:
                    scale_factor = max_size / max(h_img, w_img)
                    new_w = int(w_img * scale_factor)
                    new_h = int(h_img * scale_factor)
                    img = cv2.resize(img, (new_w, new_h))
                
                # Create mesh plane at image plane position
                # Use vertices 1,2,3,4 (the image plane rectangle)
                plane_vertices = vertices[1:5]  # bottom-left, bottom-right, top-right, top-left
                
                # Create triangles for the plane (two triangles to form rectangle)
                triangles = np.array([
                    [0, 1, 2],  # First triangle
                    [0, 2, 3]   # Second triangle
                ])
                
                # Create mesh
                image_plane = o3d.geometry.TriangleMesh()
                image_plane.vertices = o3d.utility.Vector3dVector(plane_vertices)
                image_plane.triangles = o3d.utility.Vector3iVector(triangles)
                
                # Set texture coordinates
                # UV coordinates: (0,0) = bottom-left, (1,1) = top-right
                uvs = np.array([
                    [0, 1],  # bottom-left
                    [1, 1],  # bottom-right
                    [1, 0],  # top-right
                    [0, 0]   # top-left
                ])
                image_plane.triangle_uvs = o3d.utility.Vector2dVector(
                    np.concatenate([uvs[[0,1,2]], uvs[[0,2,3]]])
                )
                
                # Convert image to Open3D format and apply texture
                try:
                    # Ensure image is contiguous and in correct format
                    img_contiguous = np.ascontiguousarray(img, dtype=np.uint8)
                    image_o3d = o3d.geometry.Image(img_contiguous)
                    
                    # Apply texture to mesh
                    image_plane.textures = [image_o3d]
                    
                    # Compute normals for proper lighting
                    image_plane.compute_vertex_normals()
                except Exception as tex_error:
                    print(f"Warning: Could not apply texture: {tex_error}")
                    # Fallback: use colored mesh instead of texture
                    image_plane = None
                
        except Exception as e:
            print(f"Warning: Could not load texture for camera at {image_path}: {e}")
    
    return line_set, image_plane


def visualize_reconstruction(reconstruction, mesh=None, use_dense=False, show_mesh=False, show_point_cloud=True, image_folder=None, load_textures=False):
    """
    Visualize 3D reconstruction with point cloud and camera poses using Open3D
    Interactive controls:
        M: Toggle mesh visibility
        P: Toggle point cloud visibility
        C: Toggle camera frustums visibility
        I: Toggle camera images visibility
    
    Args:
        reconstruction: pycolmap reconstruction object
        mesh: Optional Open3D TriangleMesh to visualize
        use_dense: If True and dense points available, visualize dense reconstruction
        show_mesh: Initial mesh visibility
        show_point_cloud: Initial point cloud visibility
        image_folder: Path to folder containing images (for textured frustums)
        load_textures: If True, load camera image textures (experimental, may cause crashes)
    """
    print("\n=== Preparing visualization ===")
    
    # Prepare point cloud
    pcd = None
    if use_dense and hasattr(reconstruction, 'dense_points'):
        print("Preparing dense point cloud...")
        points = reconstruction.dense_points
        colors = reconstruction.dense_colors
        mode = "Dense"
    else:
        print("Preparing sparse point cloud...")
        points, colors = reconstruction_to_arrays(reconstruction)
        mode = "Sparse"
    
    print(f"  Point cloud: {len(points)} points ({mode})")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Prepare mesh
    if mesh is not None:
        print(f"Preparing mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
        mesh.compute_vertex_normals()
    
    # Prepare camera frustums
    print("Preparing camera frustums...")
    camera_centers, camera_rotations = get_camera_positions(reconstruction)
    print(f"  {len(camera_centers)} cameras")
    
    # Calculate appropriate frustum scale
    if hasattr(reconstruction, 'dense_points'):
        ref_points = reconstruction.dense_points
    else:
        ref_points = points
    scale = np.percentile(np.linalg.norm(ref_points - np.mean(ref_points, axis=0), axis=1), 10) * 0.2
    
    camera_frustums = []
    camera_images = []
    
    # Create mapping of image names to paths
    image_path_map = {}
    if image_folder is not None:
        image_folder = Path(image_folder)
        if image_folder.exists():
            for img_path in image_folder.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image_path_map[img_path.name] = str(img_path)
    
    for idx, (center, R) in enumerate(zip(camera_centers, camera_rotations)):
        # Get image name for this camera
        image_id = list(reconstruction.images.keys())[idx]
        image_name = reconstruction.images[image_id].name
        image_path = image_path_map.get(image_name)
        
        frustum, image_plane = create_camera_frustum(center, R, scale=scale, color=[1, 0, 0], image_path=image_path, load_texture=load_textures)
        camera_frustums.append(frustum)
        if image_plane is not None:
            camera_images.append(image_plane)
    
    if camera_images:
        print(f"  Loaded {len(camera_images)} camera images as textures")
    
    # Coordinate frame
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=scale * 2, origin=[0, 0, 0]
    )
    
    # Print reconstruction statistics
    print("\n=== Reconstruction Statistics ===")
    print(f"Number of cameras: {len(reconstruction.cameras)}")
    print(f"Number of images: {len(reconstruction.images)}")
    print(f"Registered images: {len(reconstruction.images)}")
    print(f"Sparse 3D points: {len(reconstruction.points3D)}")
    if hasattr(reconstruction, 'dense_points'):
        print(f"Dense 3D points: {len(reconstruction.dense_points)}")
    if mesh is not None:
        print(f"Mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
    
    # Create custom visualizer with key callbacks
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Structure from Motion - Interactive", width=1280, height=960, left=50, top=50)
    
    # State tracking
    state = {
        'mesh_visible': show_mesh and mesh is not None,
        'pcd_visible': show_point_cloud,
        'cameras_visible': True,
        'images_visible': True,
        'mesh': mesh,
        'pcd': pcd,
        'cameras': camera_frustums,
        'images': camera_images,
        'coord_frame': coord_frame,
        'vis': vis
    }
    
    # Add initial geometries
    if state['pcd_visible']:
        vis.add_geometry(pcd)
    if state['mesh_visible']:
        vis.add_geometry(mesh)
    if state['cameras_visible']:
        for frustum in camera_frustums:
            vis.add_geometry(frustum)
    if state['images_visible']:
        for img_plane in camera_images:
            vis.add_geometry(img_plane)
    vis.add_geometry(coord_frame)
    
    # Toggle mesh visibility (M key)
    def toggle_mesh(vis):
        if state['mesh'] is None:
            print("No mesh available")
            return False
        
        if state['mesh_visible']:
            vis.remove_geometry(state['mesh'], reset_bounding_box=False)
            state['mesh_visible'] = False
            print("Mesh: Hidden")
        else:
            vis.add_geometry(state['mesh'], reset_bounding_box=False)
            state['mesh_visible'] = True
            print("Mesh: Visible")
        return False
    
    # Toggle point cloud visibility (P key)
    def toggle_pointcloud(vis):
        if state['pcd_visible']:
            vis.remove_geometry(state['pcd'], reset_bounding_box=False)
            state['pcd_visible'] = False
            print("Point Cloud: Hidden")
        else:
            vis.add_geometry(state['pcd'], reset_bounding_box=False)
            state['pcd_visible'] = True
            print("Point Cloud: Visible")
        return False
    
    # Toggle camera frustums (C key)
    def toggle_cameras(vis):
        if state['cameras_visible']:
            for frustum in state['cameras']:
                vis.remove_geometry(frustum, reset_bounding_box=False)
            state['cameras_visible'] = False
            print("Cameras: Hidden")
        else:
            for frustum in state['cameras']:
                vis.add_geometry(frustum, reset_bounding_box=False)
            state['cameras_visible'] = True
            print("Cameras: Visible")
        return False
    
    # Toggle camera images (I key)
    def toggle_images(vis):
        if len(state['images']) == 0:
            print("No camera images available")
            return False
        
        if state['images_visible']:
            for img_plane in state['images']:
                vis.remove_geometry(img_plane, reset_bounding_box=False)
            state['images_visible'] = False
            print("Camera Images: Hidden")
        else:
            for img_plane in state['images']:
                vis.add_geometry(img_plane, reset_bounding_box=False)
            state['images_visible'] = True
            print("Camera Images: Visible")
        return False
    
    # Register key callbacks
    vis.register_key_callback(ord("M"), toggle_mesh)
    vis.register_key_callback(ord("P"), toggle_pointcloud)
    vis.register_key_callback(ord("C"), toggle_cameras)
    vis.register_key_callback(ord("I"), toggle_images)
    
    # Get render options and view control
    render_option = vis.get_render_option()
    render_option.point_size = 2.0
    render_option.mesh_show_back_face = False
    
    # Set initial camera view right behind first camera
    view_control = vis.get_view_control()
    if len(camera_centers) > 0:
        # Position viewer right behind the first camera, looking in the same direction
        first_cam_center = camera_centers[0]
        first_cam_rotation = camera_rotations[0]
        
        # Camera looks along its -Z axis in world coordinates (COLMAP convention)
        view_distance = scale * 1.5  # Distance behind the camera
        camera_forward = first_cam_rotation @ np.array([0, 0, -1])  # Camera's forward direction (looking direction)
        
        # Position viewer behind camera
        viewer_position = first_cam_center - camera_forward * view_distance
        
        # Look straight ahead in the camera's direction
        lookat_point = first_cam_center + camera_forward * view_distance
        
        view_control.set_lookat(lookat_point)
        view_control.set_front(camera_forward)
        view_control.set_up(first_cam_rotation @ np.array([0, -1, 0]))  # Camera's up direction
        view_control.set_zoom(0.7)
    
    print("\n=== Interactive Viewer Controls ===")
    print("  Mouse Left + Drag: Rotate view")
    print("  Mouse Middle + Drag (or Shift + Left): Translate view")
    print("  Mouse Wheel: Zoom in/out")
    print("  Ctrl + Left Mouse: Change field of view")
    print("  R: Reset camera view")
    print("  M: Toggle mesh visibility")
    print("  P: Toggle point cloud visibility")
    print("  C: Toggle camera frustums visibility")
    print("  I: Toggle camera images visibility")
    print("  H: Print help")
    print("  Q/ESC: Close window")
    print("\nOpening viewer...")
    print("\nNote: On macOS, use Shift + Left Mouse Drag to translate")
    
    # Run visualizer with proper event loop
    vis.run()
    vis.destroy_window()


def main():
    print("=== Structure from Motion with pycolmap ===\n")
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Structure from Motion reconstruction with pycolmap')
    parser.add_argument('--dense', action='store_true', help='Enable dense reconstruction')
    parser.add_argument('--dense-method', choices=['colmap', 'stereo', 'openmvs'], default='stereo',
                        help='Dense reconstruction method (default: stereo)')
    parser.add_argument('--mesh', action='store_true', default=True, help='Generate mesh (default: True)')
    parser.add_argument('--no-mesh', dest='mesh', action='store_false', help='Disable mesh generation')
    parser.add_argument('--mesh-method', choices=['poisson', 'ball_pivoting'], default='poisson',
                        help='Mesh generation method (default: poisson)')
    parser.add_argument('--show-mesh', action='store_true', default=True, help='Show mesh in viewer (default: True)')
    parser.add_argument('--no-show-mesh', dest='show_mesh', action='store_false', help='Hide mesh in viewer')
    parser.add_argument('--show-points', action='store_true', default=True, help='Show point cloud (default: True)')
    parser.add_argument('--no-show-points', dest='show_points', action='store_false', help='Hide point cloud')
    parser.add_argument('--force-recompute', action='store_true', help='Delete cached data and recompute from scratch')
    parser.add_argument('--camera-images', action='store_true', help='Show camera images as textures (experimental, may crash on some systems)')
    parser.add_argument('--image-folder', default='resource/sfm_images', help='Path to input images')
    parser.add_argument('--output-folder', default='resource/sfm_output', help='Path to output directory')
    
    args = parser.parse_args()
    
    # Configuration from command-line arguments
    DENSE_RECONSTRUCTION = args.dense
    DENSE_METHOD = args.dense_method
    GENERATE_MESH = args.mesh
    MESH_METHOD = args.mesh_method
    SHOW_MESH = args.show_mesh
    SHOW_POINT_CLOUD = args.show_points
    FORCE_RECOMPUTE = args.force_recompute
    
    # Display configuration
    print("Configuration:")
    print(f"  Dense reconstruction: {DENSE_RECONSTRUCTION}")
    if DENSE_RECONSTRUCTION:
        print(f"  Dense method: {DENSE_METHOD}")
    print(f"  Generate mesh: {GENERATE_MESH}")
    if GENERATE_MESH:
        print(f"  Mesh method: {MESH_METHOD}")
    print(f"  Show mesh: {SHOW_MESH}")
    print(f"  Show point cloud: {SHOW_POINT_CLOUD}")
    print(f"  Force recompute: {FORCE_RECOMPUTE}")
    print()
    
    # Visualization combinations:
    # - SHOW_MESH=True, SHOW_POINT_CLOUD=True: Shows both mesh and points (overlay)
    # - SHOW_MESH=True, SHOW_POINT_CLOUD=False: Shows only mesh
    # - SHOW_MESH=False, SHOW_POINT_CLOUD=True: Shows only point cloud
    # - Both False: Shows only cameras (not recommended)
    
    # Note: Dense reconstruction options:
    # - 'colmap': High quality but requires CUDA/GPU (COLMAP's patch_match_stereo)
    # - 'openmvs': High quality CPU-based reconstruction (requires OpenMVS installation)
    #              Install: brew install openmvs (macOS) or build from source
    #              https://github.com/cdcseacave/openMVS
    # - 'stereo': Simple OpenCV stereo matching (experimental, limited results)
    
    # Note: Mesh generation methods:
    # - 'poisson': Smooth, watertight mesh (recommended for most cases)
    # - 'ball_pivoting': Preserves details but may have holes
    
    # Define paths
    image_folder = args.image_folder
    output_folder = args.output_folder
    
    # Check if image folder exists
    if not os.path.exists(image_folder):
        print(f"Image folder '{image_folder}' not found.")
        print("Creating example folder structure...")
        os.makedirs(image_folder, exist_ok=True)
        print(f"\nPlease place your image sequence in: {image_folder}")
        print("Then run this script again.")
        return
    
    # Load images to verify
    images = load_images_from_folder(image_folder)
    
    if len(images) < 2:
        print(f"\nError: Need at least 2 images for reconstruction. Found {len(images)}")
        print(f"Please add more images to: {image_folder}")
        return
    
    # Display info about images
    print(f"\n=== Image Set Info ===")
    print(f"First image: {Path(images[0][0]).name}")
    print(f"Last image: {Path(images[-1][0]).name}")
    print(f"Image dimensions: {images[0][1].shape[1]}x{images[0][1].shape[0]}")
    
    try:
        # Run SfM reconstruction
        reconstruction, mesh, openmvs_mesh_path = run_sfm_reconstruction(
            image_folder, output_folder, 
            dense=DENSE_RECONSTRUCTION,
            dense_method=DENSE_METHOD,
            generate_mesh=GENERATE_MESH,
            mesh_method=MESH_METHOD,
            force_recompute=FORCE_RECOMPUTE
        )
        
        if reconstruction is not None:
            # Extract and save SIFT descriptors for tracking
            output_path = Path(output_folder)
            database_path = output_path / "database.db"
            extract_and_save_descriptors(database_path, reconstruction, output_path / "0")
            
            # Visualize results
            use_dense = DENSE_RECONSTRUCTION and hasattr(reconstruction, 'dense_points')
            show_mesh = GENERATE_MESH and SHOW_MESH and mesh is not None
            visualize_reconstruction(
                reconstruction, 
                mesh=mesh, 
                use_dense=use_dense, 
                show_mesh=show_mesh,
                show_point_cloud=SHOW_POINT_CLOUD,
                image_folder=image_folder,
                load_textures=args.camera_images
            )
            
            # Export point cloud to PLY
            output_path = Path(output_folder)
            if use_dense:
                points = reconstruction.dense_points
                colors = reconstruction.dense_colors
                ply_path = output_path / "point_cloud_dense.ply"
            else:
                points, colors = reconstruction_to_arrays(reconstruction)
                ply_path = output_path / "point_cloud_sparse.ply"
            
            save_point_cloud_ply(ply_path, points, colors)
            print(f"\nPoint cloud saved to: {ply_path}")
        else:
            print("\nReconstruction failed. Please check:")
            print("1. Images have sufficient overlap")
            print("2. Images contain textured features")
            print("3. Images are not blurry")
            
    except Exception as e:
        print(f"\nError during reconstruction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
