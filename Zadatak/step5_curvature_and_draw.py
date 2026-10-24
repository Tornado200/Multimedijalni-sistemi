import sys
import cv2
import numpy as np
from common import (find_lane_pixels, robust_fit, get_src_dst, undistort,
                    OUT, REF_W, LANE_WIDTH_M, LANE_WIDTH_PX)


def measure_curvature_and_draw(original_image_path, warped_image_path):
    original_img = cv2.imread(original_image_path)
    warped_img = cv2.imread(warped_image_path, cv2.IMREAD_GRAYSCALE)
    if original_img is None or warped_img is None:
        print("Error: Could not load images for step 5.")
        return None

    original_img = undistort(original_img)
    _, binary = cv2.threshold(warped_img, 127, 255, cv2.THRESH_BINARY)

    leftx, lefty, rightx, righty, _ = find_lane_pixels(binary)
    if len(leftx) == 0 or len(rightx) == 0:
        print("Warning: Not enough pixels to calculate curvature.")
        return None

    h, w = binary.shape[:2]
    thresh = 12 * w / REF_W

    left_fit, left_mask = robust_fit(lefty, leftx, thresh)
    right_fit, right_mask = robust_fit(righty, rightx, thresh)

    ym_per_pix = 30 / h
    xm_per_pix = LANE_WIDTH_M / (LANE_WIDTH_PX * w / REF_W)

    ploty = np.linspace(0, h - 1, h)
    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(lefty[left_mask] * ym_per_pix, leftx[left_mask] * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty[right_mask] * ym_per_pix, rightx[right_mask] * xm_per_pix, 2)

    eps = 1e-9
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / max(abs(2 * left_fit_cr[0]), eps)
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / max(abs(2 * right_fit_cr[0]), eps)
    average_curverad = (left_curverad + right_curverad) / 2
    print(f"Radius of Curvature: {average_curverad:.2f} m")

    camera_center = original_img.shape[1] / 2
    left_lane_bottom = np.polyval(left_fit, y_eval)
    right_lane_bottom = np.polyval(right_fit, y_eval)
    lane_center = (left_lane_bottom + right_lane_bottom) / 2
    center_offset = (camera_center - lane_center) * xm_per_pix
    print(f"Vehicle offset from center: {center_offset:.2f} m")

    warp_zero = np.zeros_like(binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_fitx = np.polyval(left_fit, ploty)
    right_fitx = np.polyval(right_fit, ploty)

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    height, width = original_img.shape[:2]
    src, dst = get_src_dst(width, height)
    Minv = cv2.getPerspectiveTransform(dst, src)

    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

    side = "left" if center_offset < 0 else "right"
    cv2.putText(result, f'Radius of Curvature: {average_curverad:.2f} m', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Vehicle is {abs(center_offset):.2f}m {side} of center', (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    output_path = f'{OUT}/output_final.jpg'
    cv2.imwrite(output_path, result)
    print(f"Final output saved successfully to '{output_path}'.")
    return result


if __name__ == "__main__":
    original = sys.argv[1] if len(sys.argv) > 1 else 'test_images/test2.jpg'
    warped = sys.argv[2] if len(sys.argv) > 2 else f'{OUT}/output_warped.jpg'
    measure_curvature_and_draw(original, warped)