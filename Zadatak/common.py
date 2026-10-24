import os
import cv2
import numpy as np
import pickle

OUT = 'output_images'
os.makedirs(OUT, exist_ok=True)

REF_W = 1280
REF_H = 720
LANE_WIDTH_M = 3.7
LANE_WIDTH_PX = 569

_CALIBRATION = None


def undistort(img):
    global _CALIBRATION
    if img.shape[1] != REF_W or img.shape[0] != REF_H:
        return img
    if _CALIBRATION is None:
        with open('calibration.p', 'rb') as f:
            _CALIBRATION = pickle.load(f)
    mtx = _CALIBRATION['mtx']
    dist = _CALIBRATION['dist']
    return cv2.undistort(img, mtx, dist, None, mtx)


def get_src_dst(w, h):
    src = np.float32([
        [w * 0.40, h * 0.63],
        [w * 0.60, h * 0.63],
        [w * 0.90, h * 0.93],
        [w * 0.10, h * 0.93]
    ])
    offset = int(250 * w / REF_W)
    dst = np.float32([
        [offset, 0],
        [w - offset, 0],
        [w - offset, h],
        [offset, h]
    ])
    return src, dst


def find_lane_pixels(binary, draw=False):
    h, w = binary.shape[:2]
    scale = w / REF_W
    margin = int(140 * scale)
    minpix = int(20 * scale)
    nwindows = 9
    edge = int(100 * scale)

    out_img = np.dstack((binary, binary, binary)) if draw else None

    histogram = np.sum(binary[h // 2:, :], axis=0)
    midpoint = histogram.shape[0] // 2
    leftx_current = int(np.argmax(histogram[edge:midpoint])) + edge
    rightx_current = int(np.argmax(histogram[midpoint:w - edge])) + midpoint

    window_height = h // nwindows
    nonzero = binary.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = h - (window + 1) * window_height
        win_y_high = h - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if draw and out_img is not None:
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

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

    return (nonzerox[left_lane_inds], nonzeroy[left_lane_inds],
            nonzerox[right_lane_inds], nonzeroy[right_lane_inds], out_img)


def robust_fit(y, x, thresh, iters=300, sample=8):
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    n = len(y)
    if n <= sample:
        return np.polyfit(y, x, 2), np.ones(n, dtype=bool)

    rng = np.random.default_rng(0)
    best_mask = None
    best_count = -1
    for _ in range(iters):
        idx = rng.choice(n, sample, replace=False)
        fit = np.polyfit(y[idx], x[idx], 2)
        mask = np.abs(np.polyval(fit, y) - x) < thresh
        count = int(mask.sum())
        if count > best_count:
            best_count = count
            best_mask = mask

    if best_count < 3:
        return np.polyfit(y, x, 2), np.ones(n, dtype=bool)
    return np.polyfit(y[best_mask], x[best_mask], 2), best_mask