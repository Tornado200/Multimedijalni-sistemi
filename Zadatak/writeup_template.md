## Writeup Template

### You use this file as a template for your writeup.

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

In `calibration.py`, I computed the camera calibration matrix and distortion coefficients using a series of chessboard calibration images located in the `camera_cal` folder. 
- **Corner Detection:** The `cv2.findChessboardCorners()` function was used to locate internal chessboard corners.
- **Matrix Calculation:** Object points (3D points in real-world space) and image points (2D points on the image plane) were accumulated and passed to `cv2.calibrateCamera()` to obtain the camera matrix and distortion coefficients.
- **Distortion Correction:** Using `cv2.undistort()`, these parameters were applied to correct lens distortion, generating verified results such as `undistort_output.png`.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate the distortion correction, I applied the camera matrix and distortion coefficients obtained from the chessboard calibration to a raw road image using `cv2.undistort()`. This removes radial and tangential lens distortion, ensuring that straight lines (like lane markings) maintain their geometric accuracy across the frame. An example of this correction can be found in our saved outputs.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In `step2_preprocessing.py`, I implemented the initial image preprocessing pipeline:
- **Grayscale Conversion:** The image is converted to grayscale using `cv2.cvtColor()` to reduce dimensionality and simplify processing.
- **Gaussian Noise Reduction:** A Gaussian Blur filter (with a 5x5 kernel) is applied to smooth out high-frequency noise and minor irregularities.
- **Region of Interest (ROI) Masking:** A polygonal mask representing the road ahead is defined using `cv2.fillPoly()` and combined via `cv2.bitwise_and()` to isolate the target road area and eliminate irrelevant background elements (sky, trees, and surrounding traffic).

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

TODO: Add your text here!!!

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

TODO: Add your text here!!!

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

TODO: Add your text here!!!

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

TODO: Add your text here!!!

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

TODO: Add your text here!!!

