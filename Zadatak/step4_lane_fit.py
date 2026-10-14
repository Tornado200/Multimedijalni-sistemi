import cv2
import numpy as np

def find_lane_pixels_and_fit(warped_image_path):
    # 1. Load the binary warped image
    img = cv2.imread(warped_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Unable to load image from path {warped_image_path}")
        return

    # Ensure the image is strictly binary
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # 2. Compute the histogram of the bottom half of the image
    histogram = np.sum(binary[binary.shape[0] // 2:, :], axis=0)
    
    # Create a color visualization image to draw on
    out_img = np.dstack((binary, binary, binary)) * 255

    # Find x-coordinates of the left and right lane bases
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # 3. Sliding windows settings (optimized for dashed/fragmented lines)
    nwindows = 9
    window_height = int(binary.shape[0] // nwindows)
    
    # Find all non-zero pixel indices in the image
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Increased margin and lower minpix threshold to handle dashed lines better
    margin = 120
    minpix = 35
    
    # Lists to store left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        # Identify window boundaries in y-direction
        win_y_low = binary.shape[0] - (window + 1) * window_height
        win_y_high = binary.shape[0] - window * window_height
        
        # Identify window boundaries in x-direction (left and right)
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        
        # Identify non-zero pixels within the windows
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If greater than minpix pixels are found, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # 4. Fit a second-order polynomial (x = ay^2 + by + c)
    if len(leftx) > 0 and len(rightx) > 0:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
    else:
        print("Warning: Not enough pixels found to fit lane lines.")
        return

    # Generate y values for plotting
    ploty = np.linspace(0, binary.shape[0] - 1, binary.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Color the pixels in the output image
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]   # Left lane - Red
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255] # Right lane - Blue

    # Save the result
    output_path = 'output_images/output_fit.jpg'
    cv2.imwrite(output_path, out_img)
    print(f"Lane detection and polynomial fitting completed successfully! Saved to '{output_path}'.")

if __name__ == "__main__":
    find_lane_pixels_and_fit('output_images/output_warped.jpg')