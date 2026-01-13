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

def detect_initial_pose(img_ref, kp_ref, des_ref, frame, detector, descriptor, norm_type,
                        ratio_threshold, ransac_threshold, min_matches,
                        camera_matrix, target_corners_3d, max_reproj_error, max_track_points):
    """
    Detect features, match them, and compute initial pose
    Returns: success, rvec, tvec, H, reproj_error, ref_track_pts_2d, ref_track_pts_3d
    """
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute features in frame
    if descriptor is None:
        kp_frame, des_frame = detector.detectAndCompute(gray_frame, None)
    else:
        kp_frame = detector.detect(gray_frame, None)
        kp_frame, des_frame = descriptor.compute(gray_frame, kp_frame)
    
    if des_ref is None or des_frame is None or len(kp_frame) == 0:
        return False, None, None, None, -1.0, None, None
    
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
    
    if len(good_matches) < min_matches:
        return False, None, None, None, -1.0, None, None
    
    # Extract matched keypoint locations
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography matrix using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
    
    if H is None or mask is None:
        return False, None, None, None, -1.0, None, None
    
    try:
        # Use homography to map reference image corners to camera frame
        h_ref, w_ref = img_ref.shape[:2]
        ref_corners_2d = np.float32([
            [0, 0],
            [w_ref, 0],
            [w_ref, h_ref],
            [0, h_ref]
        ]).reshape(-1, 1, 2)
        
        # Transform reference corners to camera frame using homography
        cam_corners_2d = cv2.perspectiveTransform(ref_corners_2d, H)
        cam_corners_2d = cam_corners_2d.reshape(-1, 2)
        
        # Solve PnP to get pose
        success, rvec, tvec = cv2.solvePnP(
            target_corners_3d,
            cam_corners_2d,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return False, None, None, None, -1.0, None, None
        
        # Calculate reprojection error
        reproj_error = calculate_reprojection_error(
            target_corners_3d, cam_corners_2d, rvec, tvec, camera_matrix
        )
        
        # Check if reprojection error is acceptable
        if reproj_error > max_reproj_error:
            print(f"Detection rejected: reproj error {reproj_error:.2f} > threshold {max_reproj_error:.2f}")
            return False, None, None, None, reproj_error, None, None
        
        # Select inlier matches for tracking
        inlier_indices = [i for i in range(len(good_matches)) if mask[i]]
        
        # Limit to max_track_points
        if len(inlier_indices) > max_track_points:
            # Randomly select subset
            selected_indices = np.random.choice(inlier_indices, max_track_points, replace=False)
        else:
            selected_indices = inlier_indices
        
        # Get 2D points in reference image for tracking
        ref_track_pts_2d = np.float32([kp_ref[good_matches[i].queryIdx].pt for i in selected_indices])
        
        # Compute corresponding 3D points on target plane
        h_ref, w_ref = img_ref.shape[:2]
        target_width = target_corners_3d[1][0] - target_corners_3d[0][0]
        target_height = target_corners_3d[2][1] - target_corners_3d[1][1]
        
        # Map each reference 2D point to 3D target plane
        ref_track_pts_3d = []
        for pt_2d in ref_track_pts_2d:
            # Normalize to [0, 1] range in reference image
            u = pt_2d[0] / w_ref
            v = pt_2d[1] / h_ref
            
            # Map to 3D target plane
            x_3d = target_corners_3d[0][0] + u * target_width
            y_3d = target_corners_3d[0][1] + v * target_height
            z_3d = 0.0  # On target plane
            
            ref_track_pts_3d.append([x_3d, y_3d, z_3d])
        
        ref_track_pts_3d = np.float32(ref_track_pts_3d)
        
        return True, rvec, tvec, H, reproj_error, ref_track_pts_2d, ref_track_pts_3d
        
    except Exception as e:
        print(f"Detection failed: {e}")
        return False, None, None, None, -1.0, None, None

def track_with_warped_klt(img_ref_gray, curr_gray, prev_H, ref_track_pts_2d, ref_track_pts_3d,
                          camera_matrix, max_reproj_error, klt_params, min_track_points,
                          target_corners_3d, max_track_points):
    """
    Track reference points using KLT with warped reference image
    1. Warp reference image by homography
    2. Track points from warped reference to current frame
    3. Update pose with tracked correspondences
    4. Replenish lost points if needed
    Returns: success, rvec, tvec, new_H, reproj_error, curr_track_pts_2d, tracked_3d_pts, new_ref_pts_2d, new_ref_pts_3d
    """
    try:
        # Warp reference image to match current perspective
        h_curr, w_curr = curr_gray.shape[:2]
        warped_ref = cv2.warpPerspective(img_ref_gray, prev_H, (w_curr, h_curr))
        
        # Transform reference tracking points to warped image coordinates
        warped_track_pts = cv2.perspectiveTransform(
            ref_track_pts_2d.reshape(-1, 1, 2).astype(np.float32), prev_H
        ).reshape(-1, 2)
        
        # Filter out points that are out of bounds in warped image
        valid_mask = (
            (warped_track_pts[:, 0] >= 0) & (warped_track_pts[:, 0] < w_curr) &
            (warped_track_pts[:, 1] >= 0) & (warped_track_pts[:, 1] < h_curr)
        )
        
        if len(valid_mask) < min_track_points:
            return False, None, None, None, -1.0, None, None, None, None
        
        warped_track_pts_valid = warped_track_pts[valid_mask]
        ref_track_pts_2d_valid = ref_track_pts_2d[valid_mask]
        ref_track_pts_3d_valid = ref_track_pts_3d[valid_mask]
        
        # Track points from warped reference to current frame using KLT
        curr_track_pts, status, error = cv2.calcOpticalFlowPyrLK(
            warped_ref, curr_gray,
            warped_track_pts_valid.reshape(-1, 1, 2).astype(np.float32),
            None,
            **klt_params
        )
        
        if curr_track_pts is None or np.sum(status) < min_track_points:
            return False, None, None, None, -1.0, None, None, None, None
        
        # Filter tracked points
        good_mask = status.ravel() == 1
        curr_track_pts_2d = curr_track_pts[status == 1].reshape(-1, 2)
        tracked_3d_pts = ref_track_pts_3d_valid[good_mask]
        
        if len(curr_track_pts_2d) < min_track_points:
            return False, None, None, None, -1.0, None, None, None, None
        
        # Check if we need to replenish points
        replenish_threshold = max_track_points * 0.5
        new_ref_pts_2d = ref_track_pts_2d_valid[good_mask]
        new_ref_pts_3d = tracked_3d_pts
        
        if len(curr_track_pts_2d) < replenish_threshold:
            # Detect new features in current frame
            mask = np.ones_like(curr_gray) * 255
            
            # Create mask to avoid detecting near existing points
            for pt in curr_track_pts_2d:
                cv2.circle(mask, (int(pt[0]), int(pt[1])), 10, 0, -1)
            
            # Detect good features to track
            new_features = cv2.goodFeaturesToTrack(
                curr_gray,
                maxCorners=max_track_points,
                qualityLevel=0.01,
                minDistance=10,
                mask=mask
            )
            
            if new_features is not None and len(new_features) > 0:
                new_features = new_features.reshape(-1, 2)
                
                # Map new features back to reference image coordinates using inverse homography
                try:
                    H_inv = np.linalg.inv(prev_H)
                    new_features_in_ref = cv2.perspectiveTransform(
                        new_features.reshape(-1, 1, 2).astype(np.float32), H_inv
                    ).reshape(-1, 2)
                    
                    # Filter points that are within reference image bounds
                    h_ref, w_ref = img_ref_gray.shape[:2]
                    valid_new = (
                        (new_features_in_ref[:, 0] >= 0) & (new_features_in_ref[:, 0] < w_ref) &
                        (new_features_in_ref[:, 1] >= 0) & (new_features_in_ref[:, 1] < h_ref)
                    )
                    
                    if np.sum(valid_new) > 0:
                        new_features_in_ref_valid = new_features_in_ref[valid_new]
                        new_features_curr_valid = new_features[valid_new]
                        
                        # Compute 3D positions for new points
                        target_width = target_corners_3d[1][0] - target_corners_3d[0][0]
                        target_height = target_corners_3d[2][1] - target_corners_3d[1][1]
                        
                        new_3d_pts = []
                        for pt_2d in new_features_in_ref_valid:
                            u = pt_2d[0] / w_ref
                            v = pt_2d[1] / h_ref
                            x_3d = target_corners_3d[0][0] + u * target_width
                            y_3d = target_corners_3d[0][1] + v * target_height
                            z_3d = 0.0
                            new_3d_pts.append([x_3d, y_3d, z_3d])
                        
                        new_3d_pts = np.float32(new_3d_pts)
                        
                        # Add new points to tracked points
                        curr_track_pts_2d = np.vstack([curr_track_pts_2d, new_features_curr_valid])
                        tracked_3d_pts = np.vstack([tracked_3d_pts, new_3d_pts])
                        new_ref_pts_2d = np.vstack([new_ref_pts_2d, new_features_in_ref_valid])
                        new_ref_pts_3d = np.vstack([new_ref_pts_3d, new_3d_pts])
                        
                        print(f"Replenished points: {len(new_features_curr_valid)} added, total: {len(curr_track_pts_2d)}")
                except Exception as e:
                    print(f"Point replenishment failed: {e}")
        
        # Solve PnP with tracked correspondences
        success, rvec, tvec = cv2.solvePnP(
            tracked_3d_pts,
            curr_track_pts_2d,
            camera_matrix,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return False, None, None, None, -1.0, None, None, None, None
        
        # Calculate reprojection error
        reproj_error = calculate_reprojection_error(
            tracked_3d_pts, curr_track_pts_2d, rvec, tvec, camera_matrix
        )
        
        if reproj_error > max_reproj_error:
            return False, None, None, None, reproj_error, None, None, None, None
        
        # Compute new homography from tracked points
        # Map current tracked points back to reference coordinates
        tracked_ref_pts = new_ref_pts_2d
        
        # Need at least 4 points for homography
        if len(tracked_ref_pts) < 4:
            return False, None, None, None, -1.0, None, None, None, None
        
        new_H, _ = cv2.findHomography(tracked_ref_pts, curr_track_pts_2d, 0)
        
        if new_H is None:
            return False, None, None, None, -1.0, None, None, None, None
        
        return True, rvec, tvec, new_H, reproj_error, curr_track_pts_2d, tracked_3d_pts, new_ref_pts_2d, new_ref_pts_3d
        
    except Exception as e:
        print(f"Warped KLT tracking failed: {e}")
        return False, None, None, None, -1.0, None, None, None, None

def draw_tracked_region(img, H, img_ref_shape, offset_x=0):
    """
    Draw the tracked region boundary
    """
    h_ref, w_ref = img_ref_shape[:2]
    ref_corners = np.float32([
        [0, 0],
        [w_ref, 0],
        [w_ref, h_ref],
        [0, h_ref]
    ]).reshape(-1, 1, 2)
    
    tracked_corners = cv2.perspectiveTransform(ref_corners, H)
    tracked_corners = tracked_corners.astype(int)
    
    # Apply offset
    tracked_corners[:, 0, 0] += offset_x
    
    # Draw boundary
    for i in range(4):
        pt1 = tuple(tracked_corners[i].ravel())
        pt2 = tuple(tracked_corners[(i + 1) % 4].ravel())
        img = cv2.line(img, pt1, pt2, (0, 255, 255), 2)  # Yellow color
    
    return img

def main():
    # Tuning parameters
    NUM_FEATURES = 1000
    RATIO_THRESHOLD = 0.70
    RANSAC_THRESHOLD = 5.0
    MIN_MATCHES = 4
    MAX_REPROJ_ERROR = 10.0
    MAX_TRACK_POINTS = 300
    MIN_TRACK_POINTS = 20
    
    # KLT parameters for affine tracking
    klt_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )
    
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
    
    print(f"\nTracking Parameters:")
    print(f"  Max track points: {MAX_TRACK_POINTS}")
    print(f"  Min track points: {MIN_TRACK_POINTS}")
    print(f"  Max reproj error: {MAX_REPROJ_ERROR} pixels")
    print(f"  Method: Warped KLT Tracking")
    
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
    print("  - Press 'r' to force re-detection")
    print(f"\nStarting camera stream ({CAMERA_WIDTH}x{CAMERA_HEIGHT})...")
    
    # Tracking state
    tracking_active = False
    prev_gray = None
    rvec, tvec = None, None
    current_H = None
    ref_track_pts_2d = None
    ref_track_pts_3d = None
    curr_track_pts_2d = None
    
    frame_count = 0
    detect_count = 0
    track_count = 0
    reproj_error = -1.0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Prepare left pane (reference image + warped reference if tracking)
        h_ref, w_ref = img_ref.shape[:2]
        h_frame, w_frame = frame.shape[:2]
        
        # Resize reference image to fit half of frame height for better display
        target_h = h_frame // 2
        scale = target_h / h_ref
        target_w = int(w_ref * scale)
        ref_resized = cv2.resize(img_ref, (target_w, target_h))
        
        if tracking_active and current_H is not None:
            # Create warped reference image
            warped_ref = cv2.warpPerspective(img_ref, current_H, (w_frame, h_frame))
            
            # Resize warped reference to match resized reference dimensions
            warped_ref_resized = cv2.resize(warped_ref, (target_w, target_h))
            
            # Stack reference and warped reference vertically
            left_pane = np.vstack([ref_resized, warped_ref_resized])
            
            # Add labels
            cv2.putText(left_pane, "Reference", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(left_pane, "Warped Ref", (10, target_h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Just show reference image (resized)
            left_pane = np.zeros((h_frame, target_w, 3), dtype=np.uint8)
            left_pane[:target_h, :target_w] = ref_resized
            cv2.putText(left_pane, "Reference", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Create display frame
        h1, w1 = left_pane.shape[:2]
        h2, w2 = frame.shape[:2]
        result = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        result[:h1, :w1] = left_pane
        result[:h2, w1:w1+w2] = frame
        
        has_pose = False
        mode_text = ""
        
        # Try tracking first if active
        if tracking_active and current_H is not None and ref_track_pts_2d is not None:
            success, rvec, tvec, current_H, reproj_error, curr_track_pts_2d, tracked_3d, ref_track_pts_2d, ref_track_pts_3d = track_with_warped_klt(
                gray_ref, curr_gray, current_H, ref_track_pts_2d, ref_track_pts_3d,
                camera_matrix, MAX_REPROJ_ERROR, klt_params, MIN_TRACK_POINTS,
                target_corners_3d, MAX_TRACK_POINTS
            )
            
            if success:
                has_pose = True
                mode_text = "TRACKING (Warped KLT)"
                track_count += 1
                
                # Draw tracked points
                for pt in curr_track_pts_2d:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(result, (w1 + x, y), 3, (0, 255, 0), -1)
                
                # Draw tracked region boundary
                result = draw_tracked_region(result, current_H, gray_ref.shape, offset_x=w1)
            else:
                # Tracking failed, switch to detection
                tracking_active = False
                current_H = None
                ref_track_pts_2d = None
                ref_track_pts_3d = None
                print("Warped KLT tracking lost, switching to detection...")
        
        # If not tracking or tracking failed, try detection
        if not tracking_active:
            success, rvec, tvec, current_H, reproj_error, ref_track_pts_2d, ref_track_pts_3d = detect_initial_pose(
                img_ref, kp_ref, des_ref, frame,
                detector, descriptor, norm_type,
                RATIO_THRESHOLD, RANSAC_THRESHOLD, MIN_MATCHES,
                camera_matrix, target_corners_3d, MAX_REPROJ_ERROR, MAX_TRACK_POINTS
            )
            
            if success:
                has_pose = True
                tracking_active = True
                mode_text = "DETECTED"
                detect_count += 1
                print(f"Detection successful: {len(ref_track_pts_2d)} points, starting warped KLT tracking")
                
                # Transform reference points to current frame for visualization
                curr_track_pts_2d = cv2.perspectiveTransform(
                    ref_track_pts_2d.reshape(-1, 1, 2), current_H
                ).reshape(-1, 2)
                
                # Draw detected points
                for pt in curr_track_pts_2d:
                    x, y = int(pt[0]), int(pt[1])
                    cv2.circle(result, (w1 + x, y), 3, (0, 255, 255), -1)
                
                # Draw detected region boundary
                result = draw_tracked_region(result, current_H, gray_ref.shape, offset_x=w1)
            else:
                mode_text = "SEARCHING"
        
        # Update previous frame
        prev_gray = curr_gray.copy()
        
        # Draw pose if available
        if has_pose and rvec is not None and tvec is not None:
            # Calculate center and axis length
            target_center = np.mean(target_corners_3d, axis=0)
            target_width = abs(target_corners_3d[1][0] - target_corners_3d[0][0])
            target_height = abs(target_corners_3d[2][1] - target_corners_3d[1][1])
            axis_length = min(target_width, target_height) / 3.0
            
            result = draw_target_rectangle(result, camera_matrix, rvec, tvec, target_corners_3d, offset_x=w1)
            result = draw_axis(result, camera_matrix, rvec, tvec, length=axis_length, offset_x=w1, center_point=target_center)
        
        # Add text overlay
        num_pts = len(curr_track_pts_2d) if curr_track_pts_2d is not None else 0
        info_text = f"{method_name} | Mode: {mode_text} | Points: {num_pts}"
        cv2.putText(result, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Determine error display color
        if reproj_error >= 0:
            if reproj_error <= MAX_REPROJ_ERROR:
                error_color = (0, 255, 0)  # Green - good
            else:
                error_color = (0, 0, 255)  # Red - rejected
            error_text = f"Reproj Error: {reproj_error:.2f}px | Max: {MAX_REPROJ_ERROR:.2f}px"
        else:
            error_color = (128, 128, 128)  # Gray - no data
            error_text = f"Reproj Error: --- | Max: {MAX_REPROJ_ERROR:.2f}px"
        
        cv2.putText(result, error_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, error_color, 2)
        
        stats_text = f"Detections: {detect_count} | Tracks: {track_count}"
        cv2.putText(result, stats_text, (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display the result
        cv2.imshow('Warped KLT Pose Estimation', result)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            # Save current frame
            os.makedirs('output', exist_ok=True)
            output_path = f'output/tight_{method_name.lower()}_{frame_count}.jpg'
            cv2.imwrite(output_path, result)
            print(f"Saved frame to {output_path}")
        elif key == ord('r'):
            # Force re-detection
            tracking_active = False
            current_H = None
            ref_track_pts_2d = None
            ref_track_pts_3d = None
            print("Forced re-detection")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession Statistics:")
    print(f"  Total frames: {frame_count}")
    print(f"  Detections: {detect_count}")
    print(f"  Tracked frames: {track_count}")
    print("Camera stream closed.")

if __name__ == "__main__":
    main()
