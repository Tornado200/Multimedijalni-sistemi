import cv2
import numpy as np
import os

def perspective_transform(image_path):
    # 1. Load the preprocessed binary image or raw image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Check the path.")
        return None

    height, width = img.shape[:2]

    # 2. Region of Interest (ROI) and blurring
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    mask = np.zeros_like(blurred)
    polygons = np.array([[
        (int(width * 0.1), height),
        (int(width * 0.42), int(height * 0.62)),
        (int(width * 0.58), int(height * 0.62)),
        (int(width * 0.9), height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, polygons, 255)
    roi_image = cv2.bitwise_and(blurred, mask)

    # 3. Define source points (Trapezoid covering the lane lines ahead accurately)
    src = np.float32([
        [width * 0.42, height * 0.63],  # Top-left 
        [width * 0.56, height * 0.63],  # Top-right 
        [width * 0.85, height * 0.92],  # Bottom-right
        [width * 0.15, height * 0.92]   # Bottom-left
    ])

    # 4. Define destination points (Rectangle for bird's-eye view)
    offset = 250  
    dst = np.float32([
        [offset, 0],                  # Top-left
        [width - offset, 0],          # Top-right
        [width - offset, height],     # Bottom-right
        [offset, height]              # Bottom-left
    ])

    # 5. Calculate perspective transform matrices
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)  

    # 6. Warp ROI image to bird's-eye view 
    warped = cv2.warpPerspective(roi_image, M, (width, height), flags=cv2.INTER_LINEAR)

    # 7. Save results
    os.makedirs("output_images", exist_ok=True)
    cv2.imwrite("output_images/output_roi.jpg", roi_image)
    cv2.imwrite("output_images/output_warped.jpg", warped)
    
    print("Perspective transform completed.")
    return warped, M, Minv

if __name__ == "__main__":
    perspective_transform('test_images/straight_lines1.jpg')