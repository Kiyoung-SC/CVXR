import cv2
import numpy as np
import os
import json

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

def load_target_size(json_path='resource/target_size.json'):
    """
    Load target size from JSON file
    Format: {"left": x1, "top": y1, "right": x2, "bottom": y2}
    """
    default_target = {
        'left': 0.0,
        'top': 0.0,
        'right': 720,
        'bottom': 540
    }
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f"Loaded target size from {json_path}")
                return data
        except Exception as e:
            print(f"Error loading target size: {e}")
            print("Using default values")
    else:
        print(f"Target size file not found: {json_path}")
        print("Using default values")
    
    return default_target

def calculate_reprojection_error(object_points, image_points, rvec, tvec, camera_matrix):
    """
    Calculate mean reprojection error for pose validation
    Returns: mean_error (in pixels)
    """
    # Project 3D points to image plane using the estimated pose
    projected_points, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, None)
    
    # Calculate Euclidean distance between observed and projected points
    projected_points = projected_points.reshape(-1, 2)
    image_points = image_points.reshape(-1, 2)
    
    errors = np.sqrt(np.sum((image_points - projected_points) ** 2, axis=1))
    mean_error = np.mean(errors)
    
    return mean_error

def draw_target_rectangle(img, camera_matrix, rvec, tvec, target_corners_3d, offset_x=0):
    """
    Draw rectangle showing the target boundary
    offset_x: horizontal offset for when image is part of a side-by-side view
    """
    # Project 3D target corners to image plane
    imgpts, _ = cv2.projectPoints(target_corners_3d, rvec, tvec, camera_matrix, None)
    imgpts = imgpts.astype(int)
    
    # Apply offset for side-by-side view
    imgpts[:, 0, 0] += offset_x
    
    # Draw rectangle connecting the four corners
    for i in range(4):
        pt1 = tuple(imgpts[i].ravel())
        pt2 = tuple(imgpts[(i + 1) % 4].ravel())
        img = cv2.line(img, pt1, pt2, (255, 255, 0), 2)  # Cyan color
    
    return img

def draw_axis(img, camera_matrix, rvec, tvec, length=0.3, offset_x=0, center_point=None):
    """
    Draw 3D coordinate axes on the image
    offset_x: horizontal offset for when image is part of a side-by-side view
    center_point: 3D point where axes should originate (if None, uses origin)
    """
    # Use center point if provided, otherwise use origin
    if center_point is None:
        center = np.array([0, 0, 0], dtype=np.float32)
    else:
        center = np.array(center_point, dtype=np.float32)
    
    # Define 3D points for axes (center and endpoints)
    axis_points = np.float32([
        center,                          # Center/Origin
        center + [length, 0, 0],         # X-axis (red)
        center + [0, length, 0],         # Y-axis (green)
        center + [0, 0, -length]         # Z-axis (blue) - negative because camera looks down -Z
    ])
    
    # Project 3D points to image plane
    imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, None)
    imgpts = imgpts.astype(int)
    
    # Apply offset for side-by-side view
    imgpts[:, 0, 0] += offset_x
    
    # Draw axes
    origin = tuple(imgpts[0].ravel())
    img = cv2.line(img, origin, tuple(imgpts[1].ravel()), (0, 0, 255), 3)  # X-axis: red
    img = cv2.line(img, origin, tuple(imgpts[2].ravel()), (0, 255, 0), 3)  # Y-axis: green
    img = cv2.line(img, origin, tuple(imgpts[3].ravel()), (255, 0, 0), 3)  # Z-axis: blue
    
    return img

def match_and_pose_estimation(img_ref, kp_ref, des_ref, frame, detector, descriptor, norm_type, 
                               ratio_threshold, ransac_threshold, min_matches,
                               camera_matrix, target_corners_3d, show_matches, max_reproj_error):
    """
    Match features and estimate camera pose from homography with reprojection error filtering
    Returns: result_image, kp_count, good_count, inlier_count, has_pose, rvec, tvec, reproj_error
    """
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute features in frame
    if descriptor is None:
        kp_frame, des_frame = detector.detectAndCompute(gray_frame, None)
    else:
        kp_frame = detector.detect(gray_frame, None)
        kp_frame, des_frame = descriptor.compute(gray_frame, kp_frame)
    
    # Initialize return values
    has_pose = False
    rvec, tvec = None, None
    reproj_error = -1.0
    inlier_matches = []
    
    # Check if we have descriptors
    if des_ref is None or des_frame is None or len(kp_frame) == 0:
        # Create side-by-side view without matches
        h1, w1 = img_ref.shape[:2]
        h2, w2 = frame.shape[:2]
        result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        result[:h1, :w1] = img_ref
        result[:h2, w1:w1+w2] = frame
        return result, 0, 0, 0, has_pose, rvec, tvec, reproj_error
    
    # Match features using BFMatcher
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(des_ref, des_frame, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    inlier_count = 0
    
    # Estimate pose if enough matches
    if len(good_matches) >= min_matches:
        # Extract matched keypoint locations
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        
        if H is not None and mask is not None:
            inlier_count = np.sum(mask)
            
            # Store inlier matches for visualization
            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
            
            try:
                # Use homography to map reference image corners to camera frame
                # Define corners of reference image in pixel coordinates
                h_ref, w_ref = img_ref.shape[:2]
                ref_corners_2d = np.float32([
                    [0, 0],
                    [w_ref, 0],
                    [w_ref, h_ref],
                    [0, h_ref]
                ]).reshape(-1, 1, 2)
                
                # Transform reference corners to camera frame using homography
                cam_corners_2d = cv2.perspectiveTransform(ref_corners_2d, H)
                
                # Reshape for solvePnP (needs Nx2 format)
                cam_corners_2d = cam_corners_2d.reshape(-1, 2)
                
                # Now use solvePnP to get accurate pose from 3D-2D correspondences
                # target_corners_3d are the known 3D coordinates of the target
                # cam_corners_2d are where they appear in the camera frame
                success, rvec, tvec = cv2.solvePnP(
                    target_corners_3d,
                    cam_corners_2d,
                    camera_matrix,
                    None,  # No distortion
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if success:
                    # Calculate reprojection error
                    reproj_error = calculate_reprojection_error(
                        target_corners_3d, cam_corners_2d, rvec, tvec, camera_matrix
                    )
                    
                    # Check if reprojection error is acceptable
                    if reproj_error <= max_reproj_error:
                        has_pose = True
                    else:
                        # Reject pose due to high reprojection error
                        has_pose = False
                        print(f"Rejected pose: reproj error {reproj_error:.2f} > threshold {max_reproj_error:.2f}")
            except Exception as e:
                # Handle any errors in pose estimation
                print(f"Warning: Pose estimation failed - {e}")
                has_pose = False
                reproj_error = -1.0
    
    # Create side-by-side view with or without match lines
    if show_matches and len(inlier_matches) > 0:
        # Draw matches with lines
        result = cv2.drawMatches(img_ref, kp_ref, frame, kp_frame, 
                                 inlier_matches, None,
                                 flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    else:
        # Create side-by-side view without match lines
        h1, w1 = img_ref.shape[:2]
        h2, w2 = frame.shape[:2]
        result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        result[:h1, :w1] = img_ref
        result[:h2, w1:w1+w2] = frame
    
    # Draw axis on the right side (camera frame) if pose is available
    if has_pose and rvec is not None and tvec is not None:
        w_ref = img_ref.shape[1]
        
        # Calculate center of target object
        target_center = np.mean(target_corners_3d, axis=0)
        
        # Calculate axis length as 1/3 of target size
        target_width = abs(target_corners_3d[1][0] - target_corners_3d[0][0])
        target_height = abs(target_corners_3d[2][1] - target_corners_3d[1][1])
        axis_length = min(target_width, target_height) / 3.0
        
        result = draw_target_rectangle(result, camera_matrix, rvec, tvec, target_corners_3d, offset_x=w_ref)
        result = draw_axis(result, camera_matrix, rvec, tvec, length=axis_length, offset_x=w_ref, center_point=target_center)
    
    return result, len(kp_frame), len(good_matches), inlier_count, has_pose, rvec, tvec, reproj_error

def main():
    # Tuning parameters
    NUM_FEATURES = 1000
    RATIO_THRESHOLD = 0.70
    RANSAC_THRESHOLD = 5.0
    MIN_MATCHES = 4
    MAX_REPROJ_ERROR = 10.0  # Maximum reprojection error in pixels
    
    # SIFT-specific parameters
    SIFT_N_OCTAVE_LAYERS = 3
    SIFT_CONTRAST_THRESHOLD = 0.04
    SIFT_EDGE_THRESHOLD = 10
    SIFT_SIGMA = 1.6
    
    # ORB-specific parameters
    ORB_SCALE_FACTOR = 1.2
    ORB_N_LEVELS = 8
    ORB_EDGE_THRESHOLD = 31
    ORB_FIRST_LEVEL = 0
    ORB_WTA_K = 2
    ORB_PATCH_SIZE = 31
    ORB_FAST_THRESHOLD = 5
    
    # Load camera intrinsics
    intrinsics = load_camera_intrinsics()
    focal_length = intrinsics['focal_length']
    cx, cy = intrinsics['principal_point']
    CAMERA_WIDTH = intrinsics['width']
    CAMERA_HEIGHT = intrinsics['height']
    
    # Build camera matrix
    camera_matrix = np.array([
        [focal_length, 0, cx],
        [0, focal_length, cy],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"\nCamera Intrinsics:")
    print(f"  Focal length: {focal_length}")
    print(f"  Principal point: ({cx}, {cy})")
    
    # Load target size
    target_size = load_target_size()
    left = target_size['left']
    top = target_size['top']
    right = target_size['right']
    bottom = target_size['bottom']
    
    print(f"\nTarget Size:")
    print(f"  Left-Top: ({left}, {top})")
    print(f"  Right-Bottom: ({right}, {bottom})")
    
    print(f"\nReprojection Error Threshold: {MAX_REPROJ_ERROR} pixels")
    
    # Define 3D coordinates of target corners (on Z=0 plane)
    target_corners_3d = np.float32([
        [left, top, 0],
        [right, top, 0],
        [right, bottom, 0],
        [left, bottom, 0]
    ])
    
    # Load reference image
    img_ref = cv2.imread('resource/match_refs.jpg')
    
    if img_ref is None:
        print("Error: Could not load reference image")
        return
    
    # Convert to grayscale for feature detection
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    
    print("\nSelect feature detection algorithm:")
    print("1. SIFT")
    print("2. ORB")
    print("3. FAST + FREAK")
    choice = input("Enter choice (1-3): ").strip()
    
    # Initialize detector based on user choice
    if choice == '1':
        method_name = 'SIFT'
        detector = cv2.SIFT_create(
            nfeatures=NUM_FEATURES,
            nOctaveLayers=SIFT_N_OCTAVE_LAYERS,
            contrastThreshold=SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=SIFT_EDGE_THRESHOLD,
            sigma=SIFT_SIGMA
        )
        descriptor = None
        norm_type = cv2.NORM_L2
        print("Using SIFT detector")
    elif choice == '2':
        method_name = 'ORB'
        detector = cv2.ORB_create(
            nfeatures=NUM_FEATURES,
            scaleFactor=ORB_SCALE_FACTOR,
            nlevels=ORB_N_LEVELS,
            edgeThreshold=ORB_EDGE_THRESHOLD,
            firstLevel=ORB_FIRST_LEVEL,
            WTA_K=ORB_WTA_K,
            patchSize=ORB_PATCH_SIZE,
            fastThreshold=ORB_FAST_THRESHOLD
        )
        descriptor = None
        norm_type = cv2.NORM_HAMMING
        print("Using ORB detector")
    elif choice == '3':
        method_name = 'FAST+FREAK'
        detector = cv2.FastFeatureDetector_create()
        descriptor = cv2.xfeatures2d.FREAK_create()
        norm_type = cv2.NORM_HAMMING
        print("Using FAST + FREAK detector")
    else:
        print("Invalid choice, using SIFT by default")
        method_name = 'SIFT'
        detector = cv2.SIFT_create(
            nfeatures=NUM_FEATURES,
            nOctaveLayers=SIFT_N_OCTAVE_LAYERS,
            contrastThreshold=SIFT_CONTRAST_THRESHOLD,
            edgeThreshold=SIFT_EDGE_THRESHOLD,
            sigma=SIFT_SIGMA
        )
        descriptor = None
        norm_type = cv2.NORM_L2
    
    # Extract features from reference image
    print("Extracting features from reference image...")
    if descriptor is None:
        kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
    else:
        kp_ref = detector.detect(gray_ref, None)
        kp_ref, des_ref = descriptor.compute(gray_ref, kp_ref)
    
    print(f"Reference image: {len(kp_ref)} keypoints detected")
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("  - Press 't' to toggle match lines display")
    print(f"\nStarting camera stream ({CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
    
    frame_count = 0
    show_matches = False  # Toggle for showing match lines
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        
        # Match features and estimate pose with reprojection error filtering
        result, kp_count, good_count, inlier_count, has_pose, rvec, tvec, reproj_error = match_and_pose_estimation(
            img_ref, kp_ref, des_ref, frame,
            detector, descriptor, norm_type,
            RATIO_THRESHOLD, RANSAC_THRESHOLD, MIN_MATCHES,
            camera_matrix, target_corners_3d, show_matches, MAX_REPROJ_ERROR
        )
        
        # Determine pose status with color coding
        if has_pose and rvec is not None and tvec is not None:
            pose_status = f"POSE: OK (err: {reproj_error:.2f}px)"
            pose_color = (0, 255, 0)  # Green
        elif reproj_error > 0:
            pose_status = f"POSE: REJECTED (err: {reproj_error:.2f}px)"
            pose_color = (0, 0, 255)  # Red
        else:
            pose_status = "POSE: ---"
            pose_color = (128, 128, 128)  # Gray
        
        # Add text overlay with statistics
        info_text = f"{method_name} | KP: {kp_count} | Good: {good_count} | Inliers: {inlier_count}"
        cv2.putText(result, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        match_display = "ON" if show_matches else "OFF"
        cv2.putText(result, f"{pose_status} | Matches: {match_display}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pose_color, 2)
        
        # Display the result
        cv2.imshow('Camera Pose Estimation with Error Control', result)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            # Save current frame
            os.makedirs('output', exist_ok=True)
            output_path = f'output/pose_err_{method_name.lower()}_{frame_count}.jpg'
            cv2.imwrite(output_path, result)
            print(f"Saved frame to {output_path}")
        elif key == ord('t'):
            # Toggle match lines display
            show_matches = not show_matches
            status = "ON" if show_matches else "OFF"
            print(f"Match lines display: {status}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera stream closed.")

if __name__ == "__main__":
    main()
