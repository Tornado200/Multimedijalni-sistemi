import cv2
import numpy as np

def process_image(image_path):
    # 1. Load the raw image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # 2. Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    
    # Convert to grayscale for gradient calculations
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 3. Apply Sobel gradient along x-axis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    abs_sobelx = np.absolute(sobelx)
    
    # Normalize with standard scaling. If Max is 0, set to 1 to avoid division by zero.
    max_sobel = np.max(abs_sobelx)
    if max_sobel == 0:
        scaled_sobel = np.zeros_like(abs_sobelx, dtype=np.uint8)
    else:
        scaled_sobel = np.uint8(255 * abs_sobelx / max_sobel)
    
    # Threshold gradient (Sobel x) - adjusted thresholds for straight roads
    sx_thresh_min = 25  # Slightly relaxed from 30
    sx_thresh_max = 180 # Slightly stricter upper bound to reduce noise
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh_min) & (scaled_sobel <= sx_thresh_max)] = 1
    
    # Threshold color channel (S channel) - relaxed to capture faded lines
    s_thresh_min = 90 # Slightly relaxed from 100
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1

    # Threshold lightness channel (L channel) - slightly stricter to reduce bright background noise
    l_thresh_min = 210 # Slightly stricter from 200
    l_thresh_max = 255
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh_min) & (l_channel <= l_thresh_max)] = 1

    # Combine color, lightness, and gradient binary thresholds
    combined_binary = np.zeros_like(s_binary)
    combined_binary[(s_binary == 1) | (sxbinary == 1) | (l_binary == 1)] = 255

    # 4. Region of Interest (ROI) Masking - tightened for straight lines to ignore background
    mask = np.zeros_like(combined_binary)
    imshape = img.shape
    # vertices adjusted to focus tighter on lane area for straight road
    vertices = np.array([[
        (int(imshape[1] * 0.15), imshape[0]), # Lower left, slightly tighter
        (int(imshape[1] * 0.45), int(imshape[0] * 0.62)), # Upper left, slightly tighter and lower
        (int(imshape[1] * 0.55), int(imshape[0] * 0.62)), # Upper right, slightly tighter and lower
        (int(imshape[1] * 0.85), imshape[0])  # Lower right, slightly tighter
    ]], dtype=np.int32)
    
    cv2.fillPoly(mask, vertices, 255)
    masked_binary = cv2.bitwise_and(combined_binary, mask)

    # Save output binary image
    output_path = 'output_images/output_binary.jpg'
    cv2.imwrite(output_path, masked_binary)
    print(f"Preprocessing completed on {image_path}. Saved to '{output_path}'.")
    return masked_binary

if __name__ == "__main__":
    process_image('test_images/straight_lines1.jpg')