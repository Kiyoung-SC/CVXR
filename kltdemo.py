import cv2
import numpy as np

def main():
    # KLT parameters
    MAX_FEATURES = 500
    QUALITY_LEVEL = 0.01
    MIN_DISTANCE = 10
    
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Parameters for good features to track
    feature_params = dict(
        maxCorners=MAX_FEATURES,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=7
    )
    
    # Open camera
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Read first frame
    ret, old_frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame")
        return
    
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    
    # Initialize tracking state
    tracking = False
    old_points = None
    colors = None
    
    print("\nControls:")
    print("  - Press 's' to start/restart tracking")
    print("  - Press 'q' to quit")
    print("\nStarting camera stream...")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # If tracking, perform optical flow
        if tracking and old_points is not None and len(old_points) > 0:
            # Calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_points, None, **lk_params
            )
            
            if new_points is not None:
                # Select good points (flatten status for proper indexing)
                good_mask = status.ravel() == 1
                good_new = new_points[status == 1]
                good_old = old_points[status == 1]
                
                # Check if we need to add more features
                if len(good_new) < MAX_FEATURES * 0.5:  # If less than 50% remain
                    print(f"Features dropped to {len(good_new)}, replenishing...")
                    
                    # Create mask to avoid detecting near existing points
                    mask = np.ones_like(frame_gray) * 255
                    if len(good_new) > 0:
                        for pt in good_new:
                            x, y = pt.ravel()
                            cv2.circle(mask, (int(x), int(y)), MIN_DISTANCE, 0, -1)
                    
                    # Detect new features
                    new_features = cv2.goodFeaturesToTrack(
                        frame_gray,
                        mask=mask,
                        **feature_params
                    )
                    
                    if new_features is not None and len(new_features) > 0:
                        # Add new features to existing ones
                        good_new = np.vstack([good_new, new_features.reshape(-1, 2)])
                        good_old = np.vstack([good_old, new_features.reshape(-1, 2)])
                        
                        # Generate colors for new features
                        new_colors = np.random.randint(0, 255, (len(new_features), 3))
                        if len(good_new) > len(new_features):
                            colors = np.vstack([colors[good_mask], new_colors])
                        else:
                            colors = new_colors
                    else:
                        colors = colors[good_mask]
                else:
                    colors = colors[good_mask]
                
                # Draw tracks
                for i, (new, old) in enumerate(zip(good_new, good_old)):
                    a, b = new.ravel()
                    c, d = old.ravel()
                    a, b, c, d = int(a), int(b), int(c), int(d)
                    
                    # Draw line showing motion
                    color = colors[i].tolist()
                    cv2.line(frame, (a, b), (c, d), color, 2)
                    
                    # Draw current point
                    cv2.circle(frame, (a, b), 3, color, -1)
                
                # Update for next iteration
                old_gray = frame_gray.copy()
                old_points = good_new.reshape(-1, 1, 2)
                
                # Display feature count
                info_text = f"Tracking: {len(good_new)} features"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                # Tracking failed completely, reset
                tracking = False
                old_points = None
                print("Tracking lost, press 's' to restart")
        else:
            # Not tracking, show message
            cv2.putText(frame, "Press 's' to start tracking", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the result
        cv2.imshow('KLT Feature Tracking', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            # Start or restart tracking
            print(f"Starting tracking with {MAX_FEATURES} features...")
            old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect features to track
            old_points = cv2.goodFeaturesToTrack(old_gray, **feature_params)
            
            if old_points is not None:
                tracking = True
                # Generate random colors for visualization
                colors = np.random.randint(0, 255, (len(old_points), 3))
                print(f"Detected {len(old_points)} features to track")
            else:
                print("No features detected, try different scene")
                tracking = False
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera stream closed.")

if __name__ == "__main__":
    main()
