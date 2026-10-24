## Writeup 

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

[image1]: ./examples/undistort_output.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./output_images/output_binary.jpg "Binary Example"
[image4]: ./output_images/output_warped.jpg "Warp Example"
[image5]: ./output_images/output_fit.jpg "Fit Visual"
[image6]: ./output_images/output_final.jpg "Output"
[video1]: ./project_video.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Project structure

| File | Purpose |
|---|---|
| `calibration.py` | Step 1: camera calibration (run once) |
| `common.py` | Shared code: source/destination points, sliding-window search, RANSAC polynomial fit |
| `step2_preprocessing.py` | Step 2: color thresholding into a binary image |
| `step3_perspective.py` | Step 3: perspective transform (birds-eye view) |
| `step4_lane_fit.py` | Step 4: lane-pixel detection and polynomial fit |
| `step5_curvature_and_draw.py` | Step 5: curvature, vehicle offset and overlay on the original image |

### How to run

All scripts use paths relative to the `Zadatak/` folder, so run them from there. Dependencies: `numpy` and `opencv-python`.

```bash
cd Zadatak
pip install numpy opencv-python

python3 calibration.py
python3 step2_preprocessing.py test_images/test1.jpg
python3 step3_perspective.py test_images/test1.jpg
python3 step4_lane_fit.py output_images/output_warped.jpg
python3 step5_curvature_and_draw.py test_images/test1.jpg 
```

Replace `test_images/test1.jpg` with any other road image. Step 4 reads the output of step 3, and step 5 reads the original image together with the output of step 3. Step 3 calls step 2 internally, so running step 3 alone is enough to produce both the binary and the warped image. All results are written to `output_images/`.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

**File:** `calibration.py`

The camera matrix and distortion coefficients are computed from the chessboard images in the `camera_cal` folder (9x6 inner corners):

- **Corner detection:** `cv2.findChessboardCorners()` finds the inner corners in every image. Images in which all 9x6 corners are not visible are skipped, and the script prints how many images were used.
- **Matrix calculation:** the detected 2D corners and the corresponding 3D object points (a flat grid with z = 0) are passed to `cv2.calibrateCamera()`, which returns the camera matrix `mtx` and the distortion coefficients `dist`.
- **Saving:** both are stored in `calibration.p` (pickle) so they can be reused without recalibrating.
- **Verification:** `cv2.undistort()` is applied to the first calibration image to check the result visually.

**Run:** `python3 calibration.py`

**Output:** the camera matrix and distortion coefficients printed in the console, the file `calibration.p`, and the corrected test image `examples/undistort_output.jpg`.

![Undistorted][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

The calibration from the previous section produces `calibration.p`, and `cv2.undistort()` with these parameters removes the radial and tangential lens distortion (see the corrected chessboard image above).

The single-image pipeline below operates directly on the road images from `test_images/`. Applying the saved calibration to the road images is not part of the current scripts.

![Road][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

**File:** `step2_preprocessing.py`, function `process_image()`

The image is converted to the HLS color space and two color masks are combined with a logical OR:

- **White lines:** lightness `L >= 190`.
- **Yellow lines:** hue between 15 and 35, saturation `S >= 80` and lightness `L >= 60`.
- **Region of interest:** the same trapezoid that is later used as the source of the perspective transform (`get_src_dst()` in `common.py`) is applied as a mask with `cv2.fillPoly()` and `cv2.bitwise_and()`. This removes the sky, trees, barriers and most of the surrounding traffic, and guarantees that the mask and the warp always cover the same area.

Gradient (Sobel) thresholds were tried first, but they picked up barriers, road seams and car edges, so the final version relies on color only.

**Run:** `python3 step2_preprocessing.py test_images/test1.jpg`

**Output:** `output_images/output_binary.jpg`, a black and white image in which white pixels are lane-line candidates.

![Binary][image3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

**File:** `step3_perspective.py`, function `perspective_transform()`; points defined in `get_src_dst()` in `common.py`

- **Binary input:** the function calls `process_image()` from step 2 and warps the resulting binary image.
- **Source and destination points:** the source is a trapezoid along the lane and the destination is a rectangle. All values are relative to the image size, so the same code works for images of different resolutions:

| Source | Destination |
|---|---|
| (0.40w, 0.63h) | (offset, 0) |
| (0.60w, 0.63h) | (w - offset, 0) |
| (0.90w, 0.93h) | (w - offset, h) |
| (0.10w, 0.93h) | (offset, h) |

  where `offset = 250 * w / 1280`.
- **Warping:** `cv2.getPerspectiveTransform()` computes the matrix `M` (and `Minv` for the inverse) and `cv2.warpPerspective()` applies it. Converging lane lines become approximately parallel.

**Run:** `python3 step3_perspective.py test_images/test1.jpg`

**Output:** `output_images/output_warped.jpg`, the binary image in birds-eye view (the binary image of step 2 is saved as well). The function also returns `warped`, `M` and `Minv`.

![Warped][image4]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

**Files:** `step4_lane_fit.py`, function `find_lane_pixels_and_fit()`; helpers `find_lane_pixels()` and `robust_fit()` in `common.py`

- **Histogram:** the warped image is thresholded at 127 (the JPEG compression of the saved image makes the values not exactly 0/255) and a histogram over its lower half gives the starting x-position of the left and right line. A margin at both image edges is excluded from the search.
- **Sliding windows:** 9 windows of height `h / 9` with a margin of 140 px (scaled with the image width). A window is re-centered on the mean x of its pixels when it contains more than 20 pixels, which also handles dashed lines.
- **Robust polynomial fit:** a second-order polynomial `x = ay^2 + by + c` is fitted to the pixels of each line with RANSAC (`robust_fit()`): 300 random 8-pixel samples are fitted, the model with the most inliers (distance below 12 px, scaled with the image width) is kept, and the final polynomial is fitted only on those inliers. Stray pixels from cars, barriers or shadows therefore do not pull the curve away from the lane.

**Run:** `python3 step4_lane_fit.py output_images/output_warped.jpg`

**Output:** `output_images/output_fit.jpg` showing the search windows (green), the inlier pixels of the left line (red) and the right line (blue) and the fitted curves (yellow).

![Fit][image5]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

**File:** `step5_curvature_and_draw.py`, function `measure_curvature_and_draw()`

The step repeats the pixel search and RANSAC fit of step 4 on the warped image and then:

- **Pixels to meters:** `ym_per_pix = 30 / h` (the visible road section is assumed to be about 30 m long) and `xm_per_pix = 3.7 / (700 * w / 1280)` (a lane is assumed to be 3.7 m wide and about 700 px wide in the birds-eye view). New polynomials are fitted in meters on the inlier pixels.
- **Radius of curvature:** `R = (1 + (2Ay + B)^2)^(3/2) / |2A|` is evaluated at the bottom of the image for both lines and averaged. A small epsilon avoids a division by zero on a perfectly straight line.
- **Vehicle offset:** the camera is assumed to be mounted at the center of the car, so the offset is the difference between the image center and the lane center at the bottom of the image, converted to meters. A negative value means the vehicle is left of the center.

**Run:** `python3 step5_curvature_and_draw.py test_images/test1.jpg output_images/output_warped.jpg`

**Output (console), for `test1.jpg`:**

```
Radius of Curvature: 1225.16 m
Vehicle offset from center: -0.18 m
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

**File:** `step5_curvature_and_draw.py` (same function as above)

The area between the two fitted curves is filled green in the birds-eye view, warped back onto the road image with `cv2.warpPerspective()` and the inverse matrix `Minv` (built from the same points as in step 3), and blended with the original image using `cv2.addWeighted()`. The radius of curvature and the vehicle offset are written on top of the image.

**Output:** `output_images/output_final.jpg`. For `test1.jpg` the radius of curvature is 1225.16 m and the vehicle is 0.18 m left of center.

![Output][image6]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

TODO: Add your text here!!!

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


**Problems faced**

- **Noisy binary image.** The first version combined Sobel gradients with S-channel and lightness thresholds. It also picked up the concrete barrier, road seams, shadows and the edges of other cars, and these pixels pulled the fitted curves away from the lane. Switching to white/yellow color thresholds and restricting the region of interest to the same trapezoid as the perspective transform removed most of this noise.
- **Dashed lines.** A dashed line has few pixels, so a plain least-squares polynomial extrapolated badly (the green area bent inward at the top or stopped before the line at the bottom). Sliding windows that re-center only when enough pixels are found, together with RANSAC, made the fit much more stable.
- **Different image sizes.** The constants (offset, window margin, minimum pixel count, RANSAC threshold, pixel-to-meter scale) were first written for 1280x720 images and failed on 960x540 images. They are now scaled with the image width, but the source trapezoid itself is tuned for the project camera.
- **Intermediate files.** Steps are connected through JPEG files, so the binary image has to be thresholded again in step 4 and 5.

**Where the pipeline will likely fail** (expected failure modes, not all of them tested)

- Strong shadows, glare or faded and worn lane markings, where the color thresholds miss the line or pick up non-lane pixels.
- Sharp curves, where a line leaves the source trapezoid, and hills, where the flat-road assumption of the perspective transform does not hold.
- Lane changes, other vehicles directly in front of the car, and roads with no or unusual markings.
- Images from a different camera (different field of view and horizon), because the source trapezoid is fixed.
- The meter conversion is approximate. It assumes a fixed lane width of about 700 px in the birds-eye view, but the transformed lane is wider, so the radius and the offset have a systematic error.

**How to make it more robust**

- Apply the saved camera calibration to every road image before thresholding.
- Combine the color thresholds with a gradient threshold, and use adaptive thresholds (for example CLAHE on the lightness channel) for shadows and different lighting.
- On video, smooth the polynomial coefficients over several frames, search around the previous fit instead of restarting with a histogram, and reject frames that fail sanity checks (lane width, parallel lines, sudden curvature change).
- Measure the lane width in the birds-eye view to compute the horizontal scale instead of using a fixed value.
- Save intermediate images as PNG, or pass them between steps as arrays.

