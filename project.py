import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_object_flann(query_image, train_image):
    """
    Feature matching using SIFT and FLANN-based matcher.
    """
    result = train_image.copy()
    
    # Convert images to grayscale
    img1_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)
    
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    
    # Lowe's ratio test
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
    
    MIN_MATCH_COUNT = 10
    
    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        h, w = img1_gray.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        
        # Draw bounding box
        result = cv2.polylines(result, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
    else:
        print(f"Not enough matches found: {len(good_matches)}/{MIN_MATCH_COUNT}")
    
    # Display results
    plt.figure(figsize=[15, 4])
    plt.subplot(131), plt.imshow(cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)), plt.title('Query Image')
    plt.subplot(132), plt.imshow(cv2.cvtColor(train_image, cv2.COLOR_BGR2RGB)), plt.title('Train Image')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Result')
    plt.show()

# Load images
i1 = cv2.imread('img/Query.jpg')  # Query image
i2 = cv2.imread('img/Train.jpg')   # Train image


if i1 is None or i2 is None:
    print("Error: One or both images could not be loaded. Check the file paths.")
else:
    find_object_flann(i1, i2)
