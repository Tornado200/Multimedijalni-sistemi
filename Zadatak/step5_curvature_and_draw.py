import cv2
import numpy as np

def measure_curvature_and_draw(original_image_path, warped_image_path):
    # 1. Load original and warped images
    original_img = cv2.imread(original_image_path)
    warped_img = cv2.imread(warped_image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None or warped_img is None:
        print("Error: Could not load images for step 5.")
        return

    _, binary = cv2.threshold(warped_img, 127, 255, cv2.THRESH_BINARY)

    # Re-extract lane pixels using histogram and sliding windows
    histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary.shape[0] // nwindows)
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 120
    minpix = 35
    
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if len(leftx) == 0 or len(rightx) == 0:
        print("Warning: Not enough pixels to calculate curvature.")
        return

    # 2. Fit second order polynomials in pixels
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 3. Convert pixels to meters (ym_per_pix and xm_per_pix)
    ym_per_pix = 30 / 720 
    xm_per_pix = 3.7 / 700 

    ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])
    
    average_curverad = (left_curverad + right_curverad) / 2
    print(f"Radius of Curvature: {average_curverad:.2f} m")

    # 4. Calculate vehicle position relative to center
    camera_center = original_img.shape[1] / 2
    left_lane_bottom = left_fit[0] * y_eval**2 + left_fit[1] * y_eval + left_fit[2]
    right_lane_bottom = right_fit[0] * y_eval**2 + right_fit[1] * y_eval + right_fit[2]
    lane_center = (left_lane_bottom + right_lane_bottom) / 2
    center_offset = (camera_center - lane_center) * xm_per_pix
    print(f"Vehicle offset from center: {center_offset:.2f} m")

    # 5. Warp back and draw the lane overlay on original image
    warp_zero = np.zeros_like(binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the green lane area
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    height, width = original_img.shape[:2]

    # MATCHED SOURCE AND DESTINATION POINTS WITH STEP 3 FOR straight_lines1.jpg
    src = np.float32([
        [width * 0.42, height * 0.63],  
        [width * 0.56, height * 0.63],  
        [width * 0.85, height * 0.92],  
        [width * 0.15, height * 0.92]   
    ])
    
    offset = 250  
    dst = np.float32([
        [offset, 0],                  
        [width - offset, 0],          
        [width - offset, height],     
        [offset, height]              
    ])

    # Inverse perspective matrix using the exact same points
    Minv = cv2.getPerspectiveTransform(dst, src)

    newwarp = cv2.warpPerspective(color_warp, Minv, (width, height)) 
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)

    cv2.putText(result, f'Radius of Curvature: {average_curverad:.2f} m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(result, f'Vehicle is {abs(center_offset):.2f}m {"left" if center_offset < 0 else "right"} of center', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    output_path = 'output_images/output_final.jpg'
    cv2.imwrite(output_path, result)
    print(f"Final output saved successfully to '{output_path}'.")

if __name__ == "__main__":
    measure_curvature_and_draw('test_images/straight_lines1.jpg', 'output_images/output_warped.jpg')