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
[image2]: ./examples/undistort_road.jpg "Road Transformed"
[image3]: ./output_images/output_binary.jpg "Binary Example"
[image4]: ./output_images/output_warped.jpg "Warp Example"
[image5]: ./output_images/output_fit.jpg "Fit Visual"
[image6]: ./output_images/output_final.jpg "Output"
[video1]: ./output_videos/project_video03.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

### Project structure

| File | Purpose |
|---|---|
| `calibration.py` | Step 1: camera calibration (run once) |
| `common.py` | Shared code: undistortion, constants, source/destination points, sliding-window search, RANSAC polynomial fit |
| `step2_preprocessing.py` | Step 2: color thresholding into a binary image |
| `step3_perspective.py` | Step 3: perspective transform (birds-eye view) |
| `step4_lane_fit.py` | Step 4: lane-pixel detection and polynomial fit |
| `step5_curvature_and_draw.py` | Step 5: curvature, vehicle offset and overlay on the original image |
| `video_pipeline.py` | Video: the same pipeline frame by frame, with lane tracking, sanity checks and smoothing |
| `main.py` | Entry point that runs the whole pipeline on one image or one video |

### How to run

All scripts use paths relative to the `Zadatak/` folder, so run them from there. Dependencies: `numpy` and `opencv-python`. The scripts are plain Python, so there is nothing to compile.

```bash
cd Zadatak
pip install numpy opencv-python
```

`calibration.py` has to be run first, because it creates `calibration.p`, which the other steps load to undistort the images:

```bash
python3 calibration.py
```

**Single image, step by step:**

```bash
python3 step2_preprocessing.py test_images/test1.jpg
python3 step3_perspective.py test_images/test1.jpg
python3 step4_lane_fit.py output_images/output_warped.jpg
python3 step5_curvature_and_draw.py test_images/test1.jpg output_images/output_warped.jpg
```

**Single image, all steps at once, through `main.py`:**

```bash
python3 main.py image --input test_images/test1.jpg
```

**Video, through `main.py`:**

```bash
python3 main.py video --input test_videos/project_video03.mp4
python3 main.py video --input test_videos/project_video03.mp4 --max-frames 150
```

The second form processes only the first 150 frames, which is useful for a quick test.

Replace `test_images/test1.jpg` or `test_videos/project_video03.mp4` with any other image or video from `test_images/` and `test_videos/`. Step 4 reads the output of step 3, and step 5 reads the original image together with the output of step 3. Step 3 calls step 2 internally, so running step 3 alone is enough to produce both the binary and the warped image. Image results are written to `output_images/`, and the result for every image in `test_images/` is also saved in `output_images/all_tests/`. Video results are written to `output_videos/`.

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

The saved calibration is applied to every road image by `undistort()` in `common.py`. It is called at the beginning of step 2 (before thresholding) and in step 5 (before the lane is drawn), so the thresholding, the perspective transform and the final overlay all work on the same corrected geometry. Images that are not 1280x720 come from a different camera and are returned unchanged, because the calibration is valid only for the camera that took the chessboard images.

The example below shows `test1.jpg` before (left) and after (right) the correction. The effect is most visible near the image borders, for example on the car hood and the barrier.

![Road Transformed][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

**File:** `step2_preprocessing.py`, functions `binarize()` and `process_image()`

The thresholding itself is implemented in `binarize()`, which works on an image in memory so that the video pipeline can reuse it. `process_image()` loads a file, undistorts it, calls `binarize()` and saves the result. The image is converted to the HLS color space, and two color masks are combined with a logical OR:

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

- **Pixels to meters:** `ym_per_pix = 30 / h` (the visible road section is assumed to be about 30 m long). The horizontal scale is `xm_per_pix = 3.7 / (569 * w / 1280)`: a lane is 3.7 m wide, and 569 px is the lane width measured in the birds-eye view of `straight_lines1.jpg` and `straight_lines2.jpg` (572 and 566 px) with the same code that is used for detection. `LANE_WIDTH_PX` is defined in `common.py`. New polynomials are fitted in meters on the inlier pixels.
- **Radius of curvature:** `R = (1 + (2Ay + B)^2)^(3/2) / |2A|` is evaluated at the bottom of the image for both lines and averaged. A small epsilon avoids a division by zero on a perfectly straight line.
- **Vehicle offset:** the camera is assumed to be mounted at the center of the car, so the offset is the difference between the image center and the lane center at the bottom of the image, converted to meters. A negative value means the vehicle is left of the center.

**Run:** `python3 step5_curvature_and_draw.py test_images/test1.jpg output_images/output_warped.jpg`

**Output (console), for `test1.jpg`:**

```
Radius of Curvature: 1251.67 m
Vehicle offset from center: -0.21 m
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

**File:** `step5_curvature_and_draw.py` (same function as above)

The area between the two fitted curves is filled green in the birds-eye view, warped back onto the undistorted road image with `cv2.warpPerspective()` and the inverse matrix `Minv` (built from the same points as in step 3), and blended with it using `cv2.addWeighted()`. The radius of curvature and the vehicle offset are written on top of the image.

**Output:** `output_images/output_final.jpg`. For `test1.jpg` the radius of curvature is 1251.67 m and the vehicle is 0.21 m left of center. The results for all other test images are in `output_images/all_tests/`.

![Output][image6]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

**Processed videos** (in the repository, folder `output_videos/`):

- [project_video01](output_videos/project_video01.mp4)
- [project_video02](output_videos/project_video02.mp4)
- [project_video03](output_videos/project_video03.mp4) (main result, linked as `[video1]` above)
- [challenge01](output_videos/challenge01.mp4)
- [challenge02](output_videos/challenge02.mp4)
- [challenge03](output_videos/challenge03.mp4)

For every video the pipeline also writes a log with one row per frame (`output_videos/<name>_log.csv`: whether the frame was accepted, the reason, the number of pixels found for each line, the lane width in pixels and the top/bottom width difference).

**File:** `video_pipeline.py` (started with `main.py video`)

**Run:**

```bash
python3 main.py video --input test_videos/project_video03.mp4
```

**Output:** `output_videos/<name>.mp4` (1280x720, the green lane with the radius of curvature, the vehicle offset and the status of the frame written on it) and `output_videos/<name>_log.csv`. The console shows the number of frames, the percentage of accepted frames, how often each reason occurred and the median lane width and top/bottom width difference.

**Every frame goes through the same steps as a single image** (undistort, color threshold, perspective transform, lane pixels and polynomial fit, curvature and offset, drawing). On top of that, the video pipeline adds the following (class `LaneTracker`):

- **Tracking:** after a successful frame, the lines in the next frame are searched only within 100 px of the previous polynomials (`search_around()`), which ignores most of the noise further away. If too few pixels are found, or after 5 frames in a row that fail, the search restarts with sliding windows.
- **Sanity check (`check()`):** a frame is accepted only if both lines were found (at least 150 pixels covering at least 150 px in height), the lane width at the bottom is within 25% of the expected width, the lines are roughly parallel (the width at the top and at the bottom of the birds-eye image differ by less than 1.0 m) and the lane did not jump by more than 0.7 m compared to the previous frames.
- **Weak line rescue (`rescue()`):** the right line is often dashed and has few pixels. If the check fails, the line with more pixels is trusted (provided it is close to its previous position) and the other one is replaced by a copy shifted by the current lane width. The status of such frames shows "copied from".
- **Smoothing:** the drawn lane is the mean of the polynomials from the last 8 accepted frames, so it does not jump from frame to frame.
- **Recovery:** if a frame is rejected, the last smoothed lane is drawn ("holding last lane"). After 25 rejected frames in a row the history is cleared and "Lane not detected" is shown.

**Two cameras:** `project_video01` and `project_video02` are 960x540 and come from a different camera than the chessboard images (and the other videos, 1280x720). For them the frame is enlarged to 1280x720, undistortion is skipped (the calibration is valid only for the first camera) and separate source points are used. The source points of both cameras are in `SRC` in `video_pipeline.py`.

**The video uses its own perspective transform.** With the trapezoid used for single images, the lane lines converged towards the top of the birds-eye view (the lane was 1.0 to 1.3 m narrower at the top than at the bottom), so most frames failed the parallel-lines check. A narrower trapezoid (source points in `SRC`, destination x from 320 to 960) reduced the median difference to 0.09 to 0.33 m. The video also uses its own horizontal scale, 3.7 m per 640 px, instead of the 569 px used for single images. For this reason the radius and the offset in videos and in single images are not directly comparable.

**Diagnostics, first 150 frames of each video:**

| Video | Camera | Accepted directly (`ok`) | One line copied from the other | Median lane width | Median top/bottom width difference |
|---|---|---|---|---|---|
| project_video01 | 960x540 | 92 | 58 | 626 px | 0.09 m |
| project_video03 | 1280x720 | 61 | 89 | 616 px | 0.33 m |
| challenge01 | 1280x720 | 3 | 147 | 600 px | 0.51 m |
| challenge03 | 1280x720 | 11 | 139 | 560 px | 2.45 m |

The percentage of "accepted" frames printed by the program is misleading, because a frame in which one line was rebuilt from the other also counts as accepted (it is 100% for the first 150 frames of all four videos above). The table therefore separates directly accepted frames from rescued ones.

**Results:**

- On `project_video01`, `project_video02` and `project_video03` the green lane follows the lane lines for most of the video. In some places it drifts slightly away from the real lane, mostly where the right line is dashed and the lane is built from the left line.
- On `challenge01` the right line is almost never fitted directly, so the drawn lane is based mostly on the left line.
- On `challenge02` and `challenge03` the lane deviates from the real one in several places, so the detection there is not good. Both are discussed below.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

**Problems faced**

- **Noisy binary image.** The first version combined Sobel gradients with S-channel and lightness thresholds. It also picked up the concrete barrier, road seams, shadows and the edges of other cars, and these pixels pulled the fitted curves away from the lane. Switching to white/yellow color thresholds and restricting the region of interest to the same trapezoid as the perspective transform removed most of this noise.
- **Dashed lines.** A dashed line has few pixels, so a plain least-squares polynomial extrapolated badly (the green area bent inward at the top or stopped before the line at the bottom). Sliding windows that re-center only when enough pixels are found, together with RANSAC, made the fit much more stable. On video the remaining weak frames are handled by rebuilding the dashed line from the solid one.
- **Different image sizes and cameras.** The constants (offset, window margin, minimum pixel count, RANSAC threshold, pixel-to-meter scale) were first written for 1280x720 images and failed on 960x540 images. They are now scaled with the image width, but the source trapezoid itself is tuned for the project camera, and the images and videos taken by the other camera are not undistorted.
- **The perspective transform is very sensitive to the source points.** The trapezoid that worked for single images made the lane lines converge in the birds-eye view on video, which was visible only in the log (a top/bottom width difference of 1.0 to 1.3 m) and made most frames fail the check. Measuring this difference instead of judging by eye made the cause clear.
- **Intermediate files.** In the single-image pipeline the steps are connected through JPEG files, so the binary image has to be thresholded again in step 4 and 5.

**Where the pipeline will likely fail** (some of these were observed, the others are expected)

- **Sharp curves (observed on `challenge03`).** The lane line leaves the fixed source trapezoid, the lines are no longer parallel in the birds-eye view (median top/bottom difference 2.3 to 2.5 m), and many frames need a rebuilt line or are rejected. The drawn lane deviates from the real lane.
- **Different camera tilt or road slope (`challenge02`).** The same fixed trapezoid is used, and the lane deviates in several places. A likely cause is that the horizon in this video is at a different height than in the other videos, but this was not verified.
- Strong shadows, glare or faded and worn lane markings, where the color thresholds miss the line or pick up non-lane pixels (not tested in isolation).
- Hills, where the flat-road assumption of the perspective transform does not hold, lane changes and other vehicles directly in front of the car (not tested).
- Images from a different camera (different field of view and horizon), because the source trapezoid is fixed and the calibration is valid only for the camera that took the chessboard images.
- The meter conversion is approximate. The horizontal scale comes from a lane width measured once, the vertical scale assumes about 30 m of visible road, and images and videos use different scales. On a dashed right line the extrapolation to the bottom of the image is less reliable, so the offset (for example 0.77 m on `test4.jpg`) and the radius are less certain than on images with two solid lines.

**How to make it more robust**

- Estimate the source points of the perspective transform for every video automatically, for example from the vanishing point of the two lines on a frame with a straight road, instead of using fixed points. This should help on `challenge02` and with different cameras.
- Move or widen the region of interest in sharp curves, so the lines do not leave the birds-eye image.
- Combine the color thresholds with a gradient threshold, and use adaptive thresholds (for example CLAHE on the lightness channel) for shadows and different lighting.
- Fit both lines together: they are parallel, so they can share the curvature and differ only in position. This is a cleaner version of the current line rescue.
- Measure the lane width in every frame from the detected lines instead of using a value measured once, so the scale adapts to the camera and the road.
- Calibrate the second camera with its own chessboard images, so its videos can be undistorted too.
- Save intermediate images as PNG, or pass them between steps as arrays.