#!/usr/bin/env python3
"""
Practice 07 Track: Real-time scene tracking using SfM reconstruction
- Load 3D points and camera parameters from practice07.py output
- Track scene in real-time using webcam
- Render 3D axes on tracked points
"""

import cv2
import numpy as np
import pycolmap
from pathlib import Path
import json
import os


def load_camera_intrinsics(json_path='resource/camera_intrinsics.json'):
    """
    Load camera intrinsic parameters from JSON file
    Default: focal_length=500, principal_point=(320, 240)
    """
    default_intrinsics = {
        'focal_length': 500.0,
        'principal_point': [320.0, 240.0],
        'width': 640,
        'height': 480
    }
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f"Loaded camera intrinsics from {json_path}")
                return data
        except Exception as e:
            print(f"Error loading camera intrinsics: {e}")
            print("Using default values")
    else:
        print(f"Camera intrinsics file not found: {json_path}")
        print("Using default values")
    
    return default_intrinsics


def load_reconstruction(reconstruction_path):
    """
    Load COLMAP reconstruction from output directory
    Returns: reconstruction object
    """
    reconstruction_path = Path(reconstruction_path)
    sparse_model = reconstruction_path / "0"
    
    if not sparse_model.exists():
        print(f"Error: Reconstruction not found at {sparse_model}")
        return None
    
    print(f"Loading reconstruction from {sparse_model}...")
    reconstruction = pycolmap.Reconstruction(str(sparse_model))
    
    print(f"Loaded {len(reconstruction.points3D)} 3D points")
    print(f"Loaded {len(reconstruction.cameras)} cameras")
    print(f"Loaded {len(reconstruction.images)} images")
    
    return reconstruction


def extract_reference_features(reconstruction, image_folder):
    """
    Load pre-computed SIFT descriptors from practice07.py output
    (pycolmap always uses SIFT for reconstruction)
    Returns: dict with reference data from all images
    """
    print(f"\nLoading pre-computed SIFT descriptors...")
    
    # Load SIFT descriptors (check both old and new filenames)
    descriptor_file = None
    metadata_file = None
    
    # Try new naming convention first
    sift_file = Path("resource/sfm_output/0/sift_descriptors.npz")
    orb_file = Path("resource/sfm_output/0/orb_descriptors.npz")
    
    if sift_file.exists():
        descriptor_file = sift_file
        metadata_file = Path("resource/sfm_output/0/sift_metadata.json")
    elif orb_file.exists():
        # Old naming from when we tried to support ORB
        descriptor_file = orb_file
        metadata_file = Path("resource/sfm_output/0/orb_metadata.json")
    else:
        print(f"Error: Descriptor file not found")
        print("Please run practice07.py first to generate descriptors!")
        return None
    
    # Load descriptors and 3D points
    data = np.load(descriptor_file)
    all_descriptors = data['descriptors']
    descriptor_indices = data['descriptor_indices']
    points_3d = data['points_3d']
    
    print(f"Loaded descriptors shape: {all_descriptors.shape}, dtype: {all_descriptors.dtype}")
    
    # CRITICAL: Ensure descriptors are float32 for FLANN matcher
    if all_descriptors.dtype != np.float32:
        print(f"Converting descriptors from {all_descriptors.dtype} to float32 for FLANN")
        all_descriptors = all_descriptors.astype(np.float32)
    else:
        print(f"Descriptors already float32, ready for FLANN")
    
    # Build descriptor_to_3d mapping
    descriptor_to_3d = {int(idx): points_3d[i] for i, idx in enumerate(descriptor_indices)}
    
    # Load metadata
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"✓ Loaded {metadata['num_descriptors']} descriptors from {metadata['num_images']} images")
        print(f"✓ Unique 3D points: {metadata['num_unique_3d_points']}")
    else:
        print(f"✓ Loaded {len(all_descriptors)} descriptors")
    
    # Get camera intrinsics from first camera
    first_camera = reconstruction.cameras[list(reconstruction.cameras.keys())[0]]
    focal_length = first_camera.focal_length
    if hasattr(first_camera, 'principal_point_x'):
        cx = first_camera.principal_point_x
        cy = first_camera.principal_point_y
    else:
        cx = first_camera.width / 2
        cy = first_camera.height / 2
    
    print(f"Camera: focal={focal_length:.1f}, principal=({cx:.1f}, {cy:.1f})")
    print(f"\n=== Reference Database Size ===")
    print(f"Total SIFT descriptors: {len(all_descriptors)}")
    print(f"Total 3D points with descriptors: {len(descriptor_to_3d)}")
    print(f"Database ready for matching")
    
    return {
        'ref_descriptors': all_descriptors,
        'descriptor_to_3d': descriptor_to_3d,
        'focal_length': focal_length,
        'principal_point': (cx, cy),
        'camera_width': first_camera.width,
        'camera_height': first_camera.height,
        'num_images': metadata['num_images'] if metadata_file.exists() else len(reconstruction.images)
    }


def save_reference_data(ref_data, output_path):
    """
    Save reference data to JSON file for future use
    """
    data = {
        'focal_length': float(ref_data['focal_length']),
        'principal_point': [float(ref_data['principal_point'][0]), 
                           float(ref_data['principal_point'][1])],
        'camera_width': int(ref_data['camera_width']),
        'camera_height': int(ref_data['camera_height']),
        'num_images': int(ref_data['num_images']),
        'num_descriptors': len(ref_data['ref_descriptors'])
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Save descriptor and 3D point mapping
    points_path = Path(output_path).parent / "reference_data.npz"
    descriptor_indices = list(ref_data['descriptor_to_3d'].keys())
    points_3d = np.array([ref_data['descriptor_to_3d'][idx] for idx in descriptor_indices])
    
    np.savez(points_path,
             descriptors=ref_data['ref_descriptors'],
             descriptor_indices=np.array(descriptor_indices),
             points_3d=points_3d)
    
    print(f"\nSaved reference data to {output_path}")
    print(f"Saved descriptor database to {points_path}")


def draw_axis(img, camera_matrix, rvec, tvec, length=1.0, center_point=None):
    """
    Draw 3D coordinate axes on the image
    """
    if center_point is None:
        center = np.array([0, 0, 0], dtype=np.float32)
    else:
        center = np.array(center_point, dtype=np.float32)
    
    # Define 3D points for axes
    axis_points = np.float32([
        center,
        center + [length, 0, 0],      # X-axis (red)
        center + [0, length, 0],      # Y-axis (green)
        center + [0, 0, length]       # Z-axis (blue)
    ])
    
    # Project to image
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, None)
    imgpts = imgpts.astype(int)
    
    # Draw axes
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # X: red
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y: green
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3)  # Z: blue
    
    return img


def track_with_sift(frame, ref_data, detector, camera_matrix, matcher,
                    ratio_threshold=0.75, ransac_threshold=4.0, min_matches=10):
    """
    Track scene using SIFT features and PnP RANSAC
    Returns: annotated frame, success flag, rvec, tvec, num_inliers
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute SIFT features in current frame
    kp_frame, des_frame = detector.detectAndCompute(gray, None)
    
    # Always show feature count in top-left
    num_features = len(kp_frame) if kp_frame is not None else 0
    cv2.putText(frame, f"Webcam SIFT: {num_features}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    if des_frame is None or len(kp_frame) < min_matches:
        cv2.putText(frame, f"Too few features: {len(kp_frame) if kp_frame else 0}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, False, None, None, 0
    
    # Ensure webcam descriptors are float32 for FLANN
    if des_frame.dtype != np.float32:
        des_frame = des_frame.astype(np.float32)
    
    # Match features with reference
    ref_descriptors = ref_data['ref_descriptors']
    descriptor_to_3d = ref_data['descriptor_to_3d']
    
    if ref_descriptors is None:
        cv2.putText(frame, "No reference descriptors", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return frame, False, None, None, 0
    
    # Subsample reference descriptors for faster matching
    # Use every 2nd descriptor (50% of data) for good balance
    ref_indices = list(descriptor_to_3d.keys())
    sampled_indices = ref_indices[::2]
    sampled_descriptors = ref_descriptors[sampled_indices]
    
    # Ensure float32 for FLANN
    if sampled_descriptors.dtype != np.float32:
        sampled_descriptors = sampled_descriptors.astype(np.float32)
    
    # Display database size for this frame
    cv2.putText(frame, f"DB: {len(sampled_descriptors)}/{len(ref_descriptors)} desc", 
               (frame.shape[1] - 280, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Debug: print descriptor info on first frame
    if not hasattr(track_with_sift, '_debug_printed'):
        print(f"\n=== Descriptor Matching Debug ===")
        print(f"Reference descriptors: shape={sampled_descriptors.shape}, dtype={sampled_descriptors.dtype}")
        print(f"Webcam descriptors: shape={des_frame.shape}, dtype={des_frame.dtype}")
        print(f"Matcher type: {type(matcher).__name__}")
        track_with_sift._debug_printed = True
    
    try:
        # Match: sampled ref_descriptors to des_frame (current webcam)
        matches = matcher.knnMatch(sampled_descriptors, des_frame, k=2)
    except Exception as e:
        print(f"\n!!! Matching error: {e}")
        print(f"Ref shape: {sampled_descriptors.shape}, dtype: {sampled_descriptors.dtype}")
        print(f"Webcam shape: {des_frame.shape}, dtype: {des_frame.dtype}")
        cv2.putText(frame, f"Matching error: {str(e)[:40]}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame, False, None, None, 0
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < min_matches:
        cv2.putText(frame, f"Too few matches: {len(good_matches)}/{min_matches}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return frame, False, None, None, 0
    
    # Extract 2D-3D correspondences from good matches
    obj_points = []  # 3D points
    img_points = []  # 2D points in current frame
    
    for m in good_matches:
        sampled_idx = m.queryIdx  # Index in sampled array
        frame_kp_idx = m.trainIdx
        
        # Get original reference descriptor index
        original_ref_idx = sampled_indices[sampled_idx]
        
        if original_ref_idx in descriptor_to_3d:
            obj_points.append(descriptor_to_3d[original_ref_idx])
            img_points.append(kp_frame[frame_kp_idx].pt)
    
    if len(obj_points) < min_matches:
        cv2.putText(frame, f"Too few 2D-3D matches: {len(obj_points)}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        return frame, False, None, None, 0
    
    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)
    
    # Solve PnP with RANSAC
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_points, img_points, camera_matrix, None,
            iterationsCount=1000,
            reprojectionError=ransac_threshold,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if success and inliers is not None:
            num_inliers = len(inliers)
            
            # Additional validation: check inlier ratio
            inlier_ratio = num_inliers / len(obj_points)
            
            # Reject if too few inliers or bad inlier ratio
            if num_inliers < min_matches:
                cv2.putText(frame, f"Too few inliers: {num_inliers}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                return frame, False, None, None, 0
            
            if inlier_ratio < 0.25:  # At least 25% inliers (balanced threshold)
                cv2.putText(frame, f"Low inlier ratio: {inlier_ratio:.2f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                return frame, False, None, None, 0
            
            # Validate pose: check if translation is reasonable
            translation_norm = np.linalg.norm(tvec)
            if translation_norm > 15.0:  # Reject if camera is too far (adjust based on scene scale)
                cv2.putText(frame, f"Invalid pose (dist={translation_norm:.1f})", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                return frame, False, None, None, 0
            
            # Draw matched points
            for i in inliers:
                pt = tuple(img_points[i[0]].astype(int))
                cv2.circle(frame, pt, 3, (0, 255, 0), -1)
            
            # Display info
            cv2.putText(frame, f"Features: {len(kp_frame)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Matches: {len(good_matches)}", (10, 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Inliers: {num_inliers} ({inlier_ratio:.1%})", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Reproj err < {ransac_threshold:.1f}px", (10, 105),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            return frame, True, rvec, tvec, num_inliers
        else:
            cv2.putText(frame, "PnP RANSAC failed", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return frame, False, None, None, 0
            
    except Exception as e:
        cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return frame, False, None, None, 0


def track_with_pnp_ransac(frame, ref_data, detector, camera_matrix, min_inliers=10):
    """
    Track scene using PnP RANSAC with detected features
    """
    return track_with_sift(frame, ref_data, detector, camera_matrix, min_matches=min_inliers)


def main():
    print("=== Practice 07 Track: Scene Tracking with SfM ===\n")
    
    # Configuration
    IMAGE_FOLDER = "resource/sfm_images"
    RECONSTRUCTION_PATH = "resource/sfm_output"
    REFERENCE_DATA_PATH = "resource/sfm_output/reference_data.json"
    
    # Check if reconstruction exists
    if not Path(RECONSTRUCTION_PATH).exists():
        print(f"Error: Reconstruction not found at {RECONSTRUCTION_PATH}")
        print("Please run practice07.py first to generate the reconstruction")
        return
    
    # Load reconstruction
    reconstruction = load_reconstruction(RECONSTRUCTION_PATH)
    if reconstruction is None:
        return
    
    # Extract reference features and 3D points
    ref_data = extract_reference_features(reconstruction, IMAGE_FOLDER)
    
    if ref_data['ref_descriptors'] is None:
        print("Error: Could not compute reference features")
        return
    
    # Save reference data
    save_reference_data(ref_data, REFERENCE_DATA_PATH)
    
    # Load camera intrinsics from JSON file (for webcam)
    print("\n" + "="*50)
    intrinsics = load_camera_intrinsics()
    focal_length = intrinsics['focal_length']
    cx, cy = intrinsics['principal_point']
    
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"\nCamera Matrix (from JSON):")
    print(camera_matrix)
    print(f"  Focal length: {focal_length}")
    print(f"  Principal point: ({cx}, {cy})")
    print("="*50)
    
    # Initialize SIFT detector (pycolmap uses SIFT, so we must too)
    print("\nInitializing SIFT detector...")
    detector = cv2.SIFT_create(
        nfeatures=500,
        contrastThreshold=0.04,
        edgeThreshold=10,
        sigma=1.6
    )
    
    # Use FLANN matcher for SIFT (float descriptors)
    print("Using SIFT + FLANN (KDTree)")
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
    search_params = dict(checks=100)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Initialize webcam
    print("Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera resolution to 640x480 (like practice06.py)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print(f"Camera resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    
    print("\n=== Controls ===")
    print("  Q or ESC: Quit")
    print("  S: Save current frame")
    print("\nStarting tracking...")
    print("Press Q or ESC in the video window to quit\n")
    
    frame_count = 0
    
    # Create window
    cv2.namedWindow('SfM Scene Tracking', cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Show progress every 30 frames
        if frame_count % 30 == 1:
            print(f"Processing frame {frame_count}...")
        
        # Track scene
        try:
            result, success, rvec, tvec, num_inliers = track_with_sift(
                frame, ref_data, detector, camera_matrix, matcher
            )
        except Exception as e:
            print(f"Error in tracking: {e}")
            import traceback
            traceback.print_exc()
            result = frame
            success = False
            rvec = tvec = None
            num_inliers = 0
        
        # Draw axes if tracking successful
        if success and rvec is not None and tvec is not None:
            # Calculate center of 3D points
            all_3d_points = np.array(list(ref_data['descriptor_to_3d'].values()))
            center_3d = np.mean(all_3d_points, axis=0)
            
            # Calculate axis length based on point cloud size
            max_range = np.max(np.ptp(all_3d_points, axis=0))
            axis_length = max_range * 0.3
            
            result = draw_axis(result, camera_matrix, rvec, tvec, 
                             length=axis_length, center_point=center_3d)
            
            cv2.putText(result, "TRACKING", (10, result.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(result, "NO TRACKING", (10, result.shape[0] - 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Display frame info
        cv2.putText(result, f"Frame: {frame_count}", (10, result.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('SfM Scene Tracking', result)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord('s'):  # Save frame
            save_path = f"tracked_frame_{frame_count:04d}.jpg"
            cv2.imwrite(save_path, result)
            print(f"Saved frame to {save_path}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nTracking stopped.")


if __name__ == "__main__":
    main()
