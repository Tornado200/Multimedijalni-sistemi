import cv2
import numpy as np

def prepare_image(image_path):
    # 1. LOAD IMAGE
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"Error: Could not load image from {image_path}. Check the path.")
        return None

    # 2. COLOR SPACE CONVERSION
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. FILTERING AND NOISE REDUCTION
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. DEFINE REGION OF INTEREST (ROI)
    mask = np.zeros_like(blurred)
    height, width = blurred.shape
    
    polygons = np.array([[
        (int(width * 0.1), height),             # Bottom-left
        (int(width * 0.45), int(height * 0.6)), # Top-left
        (int(width * 0.55), int(height * 0.6)), # Top-right
        (int(width * 0.9), height)              # Bottom-right
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, polygons, 255)
    roi_image = cv2.bitwise_and(blurred, mask)

    # 5. SAVE RESULTS TO DISK (Bypassing GUI imshow crash)
    cv2.imwrite("output_images/output_gray.jpg", gray)
    cv2.imwrite("output_images/output_blurred.jpg", blurred)
    cv2.imwrite("output_images/output_mask.jpg", mask)
    cv2.imwrite("output_images/output_roi.jpg", roi_image)
    
    print("Processing finished successfully! Check 'output_images' folder for output files.")
    
    return roi_image

if __name__ == "__main__":
    image_path = "test_images/solidYellowLeft.jpg"
    prepared_img = prepare_image(image_path)