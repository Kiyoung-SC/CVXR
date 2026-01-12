import cv2
import numpy as np
import os

def test_feature_matching(img_ref, img_test, gray_ref, gray_test, method_name, detector, descriptor=None, norm_type=cv2.NORM_L2, ratio_threshold=0.70, ransac_threshold=5.0, min_matches=4):
    """
    Test feature matching with a specific detector/descriptor combination
    
    Parameters:
    - ratio_threshold: Lowe's ratio test threshold
    - ransac_threshold: RANSAC reprojection threshold in pixels
    - min_matches: Minimum number of matches required to compute homography
    """
    print(f"\n{'='*60}")
    print(f"Testing {method_name}")
    print(f"{'='*60}")
    
    # Detect and compute features
    if descriptor is None:
        # Detector and descriptor are the same (e.g., SIFT, ORB)
        kp_ref, des_ref = detector.detectAndCompute(gray_ref, None)
        kp_test, des_test = detector.detectAndCompute(gray_test, None)
    else:
        # Separate detector and descriptor (e.g., FAST + FREAK)
        kp_ref = detector.detect(gray_ref, None)
        kp_test = detector.detect(gray_test, None)
        kp_ref, des_ref = descriptor.compute(gray_ref, kp_ref)
        kp_test, des_test = descriptor.compute(gray_test, kp_test)
    
    print(f"Reference image: {len(kp_ref)} keypoints detected")
    print(f"Test image: {len(kp_test)} keypoints detected")
    
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
    
    print(f"Good matches after ratio test: {len(good_matches)} (ratio={ratio_threshold})")
    
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
            
            print(f"Inlier matches after RANSAC: {len(inlier_matches)} (threshold={ransac_threshold}px)")
            
            # Draw matches
            img_matches = cv2.drawMatches(img_ref, kp_ref, img_test, kp_test, 
                                           inlier_matches, None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            # Save the result
            os.makedirs('output', exist_ok=True)
            output_path = f'output/output_matches_{method_name.lower().replace(" ", "_")}.jpg'
            cv2.imwrite(output_path, img_matches)
            print(f"Matches saved to {output_path}")
            
            return len(kp_ref), len(kp_test), len(good_matches), len(inlier_matches)
        else:
            print("Failed to compute homography")
            return len(kp_ref), len(kp_test), len(good_matches), 0
    else:
        print("Not enough good matches to compute homography")
        return len(kp_ref), len(kp_test), len(good_matches), 0

def main():
    # Tuning parameters
    NUM_FEATURES = 5000          # Number of features to detect
    RATIO_THRESHOLD = 0.70       # Lowe's ratio test threshold
    RANSAC_THRESHOLD = 5.0       # RANSAC reprojection threshold in pixels
    MIN_MATCHES = 4              # Minimum matches required for homography
    
    # Load two images
    img_ref = cv2.imread('resource/match_refs.jpg')
    img_test = cv2.imread('resource/match_test.jpg')
    
    if img_ref is None or img_test is None:
        print("Error: Could not load one or both images")
        return
    
    # Convert to grayscale for feature detection
    gray_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)
    
    print(f"Tuning Parameters:")
    print(f"  - Number of features: {NUM_FEATURES}")
    print(f"  - Ratio threshold: {RATIO_THRESHOLD}")
    print(f"  - RANSAC threshold: {RANSAC_THRESHOLD}px")
    print(f"  - Min matches: {MIN_MATCHES}")
    
    results = {}
    
    # Test 1: SIFT
    sift = cv2.SIFT_create(nfeatures=NUM_FEATURES)
    results['SIFT'] = test_feature_matching(
        img_ref, img_test, gray_ref, gray_test,
        'SIFT', sift, norm_type=cv2.NORM_L2, 
        ratio_threshold=RATIO_THRESHOLD, ransac_threshold=RANSAC_THRESHOLD, min_matches=MIN_MATCHES
    )
    
    # Test 2: ORB
    orb = cv2.ORB_create(nfeatures=NUM_FEATURES)
    results['ORB'] = test_feature_matching(
        img_ref, img_test, gray_ref, gray_test,
        'ORB', orb, norm_type=cv2.NORM_HAMMING, 
        ratio_threshold=RATIO_THRESHOLD, ransac_threshold=RANSAC_THRESHOLD, min_matches=MIN_MATCHES
    )
    
    # Test 3: FAST + FREAK
    fast = cv2.FastFeatureDetector_create()
    freak = cv2.xfeatures2d.FREAK_create()
    results['FAST+FREAK'] = test_feature_matching(
        img_ref, img_test, gray_ref, gray_test,
        'FAST FREAK', fast, descriptor=freak, norm_type=cv2.NORM_HAMMING, 
        ratio_threshold=RATIO_THRESHOLD, ransac_threshold=RANSAC_THRESHOLD, min_matches=MIN_MATCHES
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<15} {'Keypoints':<12} {'Good':<8} {'Inliers':<8}")
    print(f"{'-'*60}")
    for method, (kp_ref, kp_test, good, inliers) in results.items():
        print(f"{method:<15} {kp_ref:>5}/{kp_test:<5} {good:>7} {inliers:>7}")

if __name__ == "__main__":
    main()
