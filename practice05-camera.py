import cv2
import numpy as np
import os

def match_features_realtime(img_ref, kp_ref, des_ref, img_test, detector, descriptor=None, norm_type=cv2.NORM_L2, ratio_threshold=0.70, ransac_threshold=5.0, min_matches=4):
    """
    Match features between reference image and test frame in real-time
    """
    # Convert test image to grayscale
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    
    # Detect and compute features in test image
    if descriptor is None:
        # Detector and descriptor are the same (e.g., SIFT, ORB)
        kp_test, des_test = detector.detectAndCompute(gray_test, None)
    else:
        # Separate detector and descriptor (e.g., FAST + FREAK)
        kp_test = detector.detect(gray_test, None)
        kp_test, des_test = descriptor.compute(gray_test, kp_test)
    
    # Check if we have descriptors - still draw side-by-side if no features
    if des_ref is None or des_test is None or len(kp_test) == 0:
        # Draw empty matches to show both images side-by-side
        img_matches = cv2.drawMatches(img_ref, kp_ref, img_test, kp_test, 
                                       [], None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return img_matches, 0, 0, 0
    
    # Match features using BFMatcher
    bf = cv2.BFMatcher(norm_type, crossCheck=False)
    matches = bf.knnMatch(des_ref, des_test, k=2)
    
    # Apply Lowe's ratio test to filter good matches
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    inlier_count = 0
    inlier_matches = []
    
    # Reject outliers using Homography with RANSAC
    if len(good_matches) >= min_matches:
        # Extract location of good matches
        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_test[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix using RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        
        if mask is not None:
            # Filter matches using the mask
            inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
            inlier_count = len(inlier_matches)
    
    # Always draw matches (even if empty) to keep reference image visible
    img_matches = cv2.drawMatches(img_ref, kp_ref, img_test, kp_test, 
                                   inlier_matches, None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return img_matches, len(kp_test), len(good_matches), inlier_count

def main():
    # Tuning parameters
    NUM_FEATURES = 1000          # Number of features to detect
    RATIO_THRESHOLD = 0.70       # Lowe's ratio test threshold
    RANSAC_THRESHOLD = 5.0       # RANSAC reprojection threshold in pixels
    MIN_MATCHES = 4              # Minimum matches required for homography
    
    # SIFT-specific parameters
    SIFT_N_OCTAVE_LAYERS = 3     # Number of layers in each octave (default: 3)
    SIFT_CONTRAST_THRESHOLD = 0.04  # Contrast threshold to filter weak features (default: 0.04)
    SIFT_EDGE_THRESHOLD = 10     # Threshold to filter edge-like features (default: 10)
    SIFT_SIGMA = 1.6             # Sigma of Gaussian applied to input image (default: 1.6)
    
    # ORB-specific parameters
    ORB_SCALE_FACTOR = 1.2       # Pyramid decimation ratio (default: 1.2)
    ORB_N_LEVELS = 8             # Number of pyramid levels (default: 8)
    ORB_EDGE_THRESHOLD = 31      # Size of border where features are not detected (default: 31)
    ORB_FIRST_LEVEL = 0          # Level of pyramid to put source image to (default: 0)
    ORB_WTA_K = 2                # Number of points for oriented BRIEF descriptor (default: 2)
    ORB_PATCH_SIZE = 31          # Size of patch used by oriented BRIEF (default: 31)
    ORB_FAST_THRESHOLD = 5      # FAST threshold (default: 20)
    
    # Load reference image
    img_ref = cv2.imread('resource/match_refs.jpg')
    
    if img_ref is None:
        print("Error: Could not load reference image")
        return
    
    # Convert to grayscale for feature detection
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    
    print("Select feature detection algorithm:")
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
    
    print("\nControls:")
    print("  - Press 'q' to quit")
    print("  - Press 's' to save current frame")
    print("\nStarting camera stream...")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        
        # Match features with current frame
        result, kp_count, good_count, inlier_count = match_features_realtime(
            img_ref, kp_ref, des_ref, frame,
            detector, descriptor, norm_type,
            RATIO_THRESHOLD, RANSAC_THRESHOLD, MIN_MATCHES
        )
        
        # Add text overlay with statistics
        info_text = f"{method_name} | KP: {kp_count} | Good: {good_count} | Inliers: {inlier_count}"
        cv2.putText(result, info_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the result
        cv2.imshow('Feature Matching - Camera', result)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            # Save current frame
            os.makedirs('output', exist_ok=True)
            output_path = f'output/camera_match_{method_name.lower()}_{frame_count}.jpg'
            cv2.imwrite(output_path, result)
            print(f"Saved frame to {output_path}")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera stream closed.")

if __name__ == "__main__":
    main()
