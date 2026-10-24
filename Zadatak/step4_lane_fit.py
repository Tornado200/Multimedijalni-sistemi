import sys
import cv2
import numpy as np
from common import find_lane_pixels, robust_fit, OUT, REF_W


def find_lane_pixels_and_fit(warped_image_path):
    img = cv2.imread(warped_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image from path {warped_image_path}")
        return None

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary, draw=True)
    if len(leftx) == 0 or len(rightx) == 0:
        print("Warning: Not enough pixels found to fit lane lines.")
        return None

    h, w = binary.shape[:2]
    thresh = 12 * w / REF_W

    left_fit, left_mask = robust_fit(lefty, leftx, thresh)
    right_fit, right_mask = robust_fit(righty, rightx, thresh)

    ploty = np.linspace(0, h - 1, h)
    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    out_img[lefty[left_mask], leftx[left_mask]] = [0, 0, 255]
    out_img[righty[right_mask], rightx[right_mask]] = [255, 0, 0]

    for fx in (left_fitx, right_fitx):
        pts = np.int32(np.column_stack((fx, ploty)))
        cv2.polylines(out_img, [pts], False, (0, 255, 255), 3)

    output_path = f'{OUT}/output_fit.jpg'
    cv2.imwrite(output_path, out_img)
    print(f"Lane detection and polynomial fitting completed successfully! Saved to '{output_path}'.")
    return left_fit, right_fit


if __name__ == "__main__":
    find_lane_pixels_and_fit(sys.argv[1] if len(sys.argv) > 1 else f'{OUT}/output_warped.jpg')