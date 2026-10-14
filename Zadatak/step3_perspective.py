import cv2
import numpy as np
import os

def perspective_transform(image_path):
    # 1. LOAD IMAGE
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Check the path.")
        return None

    height, width = img.shape[:2]

    # 2. STEP 2 PREPROCESSING & ROI MASK 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    mask = np.zeros_like(blurred)
    polygons = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.45), int(height * 0.6)),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygons, 255)
    roi_image = cv2.bitwise_and(blurred, mask)

    # 3. DEFINE SOURCE POINTS (Trapezoid covering the lane lines ahead)
    src = np.float32([
        [width * 0.43, height * 0.65],  # Top-left
        [width * 0.57, height * 0.65],  # Top-right
        [width * 0.9, height * 0.9],    # Bottom-right
        [width * 0.1, height * 0.9]     # Bottom-left
    ])

    # 4. DEFINE DESTINATION POINTS (Rectangle for bird's-eye view)
    offset = 200  
    dst = np.float32([
        [offset, 0],                  # Top-left
        [width - offset, 0],          # Top-right
        [width - offset, height],     # Bottom-right
        [offset, height]              # Bottom-left
    ])

    # 5. CALCULATE PERSPECTIVE TRANSFORM MATRIX
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)  

    # 6. WARP ROI IMAGE TO BIRD'S-EYE VIEW 
    warped = cv2.warpPerspective(roi_image, M, (width, height), flags=cv2.INTER_LINEAR)

    # 7. SAVE RESULTS
    os.makedirs("output_images", exist_ok=True)
    cv2.imwrite("output_images/output_roi.jpg", roi_image)
    cv2.imwrite("output_images/output_warped.jpg", warped)
    
    print("Perspective transform completed.")
    
    return warped, M, Minv

if __name__ == "__main__":
    image_path = "test_images/solidYellowLeft.jpg" 
    perspective_transform(image_path)