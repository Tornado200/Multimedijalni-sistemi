import os
import csv
import argparse
import glob
from collections import deque, Counter

import cv2
import numpy as np

from common import REF_W, REF_H, LANE_WIDTH_M, undistort, find_lane_pixels, robust_fit
from step2_preprocessing import binarize

VIDEO_DIR = 'test_videos'
VIDEO_OUT = 'output_videos'

SRC = {
    'A': np.float32([[585, 460], [695, 460], [1127, 720], [203, 720]]),
    'B': np.float32([[553, 467], [737, 467], [1153, 720], [208, 720]]),
}
DST = np.float32([[320, 0], [960, 0], [960, REF_H], [320, REF_H]])
LANE_PX = 640
YM_PER_PIX = 30 / REF_H

HISTORY = 8
MIN_PIXELS = 150
MIN_SPAN = 150
RANSAC_THRESH = 12
SEARCH_MARGIN = 100
WIDTH_TOL = 0.25
MAX_TOP_BOTTOM_DIFF_M = 1.0
MAX_JUMP_M = 0.7
RESEARCH_AFTER = 5
RESET_AFTER = 25


def camera_of(width, height):
    return 'A' if (width, height) == (REF_W, REF_H) else 'B'


def prepare(frame):
    frame = undistort(frame)
    if frame.shape[1] != REF_W or frame.shape[0] != REF_H:
        frame = cv2.resize(frame, (REF_W, REF_H))
    return frame


def make_roi(camera):
    roi = np.zeros((REF_H, REF_W), dtype=np.uint8)
    cv2.fillPoly(roi, [SRC[camera].astype(np.int32)], 255)
    return roi


def search_around(binary, left_fit, right_fit):
    nonzeroy, nonzerox = binary.nonzero()
    left_center = np.polyval(left_fit, nonzeroy)
    right_center = np.polyval(right_fit, nonzeroy)
    left_mask = np.abs(nonzerox - left_center) < SEARCH_MARGIN
    right_mask = np.abs(nonzerox - right_center) < SEARCH_MARGIN
    return (nonzerox[left_mask], nonzeroy[left_mask],
            nonzerox[right_mask], nonzeroy[right_mask])


def fit_one(x, y):
    if len(x) < MIN_PIXELS or np.ptp(y) < MIN_SPAN:
        return None
    fit, _ = robust_fit(y, x, RANSAC_THRESH)
    return fit


class LaneTracker:
    def __init__(self, lane_px):
        self.lane_px = lane_px
        self.width_px = float(lane_px)
        self.xm = LANE_WIDTH_M / lane_px
        self.left_hist = deque(maxlen=HISTORY)
        self.right_hist = deque(maxlen=HISTORY)
        self.misses = 0

    def smoothed(self):
        if not self.left_hist:
            return None
        return np.mean(self.left_hist, axis=0), np.mean(self.right_hist, axis=0)

    def detect(self, binary):
        prev = self.smoothed()
        if prev is not None and self.misses < RESEARCH_AFTER:
            lx, ly, rx, ry = search_around(binary, prev[0], prev[1])
            if min(len(lx), len(rx)) >= MIN_PIXELS:
                return lx, ly, rx, ry
        lx, ly, rx, ry, _ = find_lane_pixels(binary)
        return lx, ly, rx, ry

    def geometry(self, left, right):
        width = np.polyval(right, REF_H - 1) - np.polyval(left, REF_H - 1)
        top = np.polyval(right, 0) - np.polyval(left, 0)
        return width, (top - width) * self.xm

    def jumped(self, left, right):
        prev = self.smoothed()
        if prev is None:
            return False
        y = REF_H - 1
        shift = max(abs(np.polyval(left, y) - np.polyval(prev[0], y)),
                    abs(np.polyval(right, y) - np.polyval(prev[1], y)))
        return shift * self.xm > MAX_JUMP_M

    def check(self, left, right):
        width, diff_m = self.geometry(left, right)
        if not (1 - WIDTH_TOL) * self.lane_px <= width <= (1 + WIDTH_TOL) * self.lane_px:
            return False, "bad width"
        if abs(diff_m) > MAX_TOP_BOTTOM_DIFF_M:
            return False, "not parallel"
        if self.jumped(left, right):
            return False, "jump"
        return True, "ok"

    def rescue(self, left, right, n_left, n_right):
        candidates = [(n, side, fit) for n, side, fit in
                      ((n_left, "left", left), (n_right, "right", right)) if fit is not None]
        if not candidates:
            return None
        n, side, fit = max(candidates, key=lambda c: c[0])

        prev = self.smoothed()
        if prev is not None:
            prev_fit = prev[0] if side == "left" else prev[1]
            y = REF_H - 1
            if abs(np.polyval(fit, y) - np.polyval(prev_fit, y)) * self.xm > MAX_JUMP_M:
                return None

        other = fit.copy()
        if side == "left":
            other[2] += self.width_px
            return fit, other, "right line copied from left"
        other[2] -= self.width_px
        return other, fit, "left line copied from right"

    def update(self, binary):
        lx, ly, rx, ry = self.detect(binary)
        left, right = fit_one(lx, ly), fit_one(rx, ry)
        stats = {"left_px": len(lx), "right_px": len(rx), "width_px": "", "diff_m": ""}

        if left is None or right is None:
            ok, reason = False, "line not found"
        else:
            width, diff_m = self.geometry(left, right)
            stats["width_px"] = round(width)
            stats["diff_m"] = round(diff_m, 2)
            ok, reason = self.check(left, right)
            if ok:
                self.width_px = 0.9 * self.width_px + 0.1 * width

        if not ok:
            rescued = self.rescue(left, right, len(lx), len(rx))
            if rescued is not None:
                left, right, reason = rescued
                ok = True

        if ok:
            self.left_hist.append(left)
            self.right_hist.append(right)
            self.misses = 0
        else:
            self.misses += 1
            if self.misses >= RESET_AFTER:
                self.left_hist.clear()
                self.right_hist.clear()
        return ok, reason, stats


def lane_metrics(left_fit, right_fit, xm):
    y_eval = REF_H - 1

    def radius(fit):
        a = fit[0] * xm / YM_PER_PIX ** 2
        b = fit[1] * xm / YM_PER_PIX
        return (1 + (2 * a * y_eval * YM_PER_PIX + b) ** 2) ** 1.5 / max(abs(2 * a), 1e-9)

    xl = np.polyval(left_fit, y_eval)
    xr = np.polyval(right_fit, y_eval)
    offset = (REF_W / 2 - (xl + xr) / 2) * xm
    return (radius(left_fit) + radius(right_fit)) / 2, offset


def draw_lane(img, left_fit, right_fit, minv):
    h, w = img.shape[:2]
    ploty = np.arange(h)
    left_pts = np.column_stack((np.polyval(left_fit, ploty), ploty))
    right_pts = np.column_stack((np.polyval(right_fit, ploty), ploty))[::-1]
    pts = np.vstack((left_pts, right_pts)).astype(np.int32)

    overlay = np.zeros_like(img)
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    back = cv2.warpPerspective(overlay, minv, (w, h))
    return cv2.addWeighted(img, 1, back, 0.3, 0)


def put_text(img, lines):
    for i, text in enumerate(lines):
        cv2.putText(img, text, (50, 50 + i * 45), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 255), 2)


def process_video(path, max_frames=None):
    os.makedirs(VIDEO_OUT, exist_ok=True)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"Error: could not open {path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    camera = camera_of(width0, height0)

    M = cv2.getPerspectiveTransform(SRC[camera], DST)
    Minv = cv2.getPerspectiveTransform(DST, SRC[camera])
    roi_mask = make_roi(camera)

    name = os.path.splitext(os.path.basename(path))[0]
    out_path = f'{VIDEO_OUT}/{name}.mp4'
    log_path = f'{VIDEO_OUT}/{name}_log.csv'
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (REF_W, REF_H))

    tracker = LaneTracker(LANE_PX)
    reasons = Counter()
    widths, diffs = [], []
    n = accepted = 0
    print(f"{name}: {width0}x{height0}, camera {camera}"
          f"{' (resized, no undistortion)' if camera == 'B' else ''}")

    with open(log_path, 'w', newline='') as f:
        log = csv.writer(f)
        log.writerow(['frame', 'accepted', 'reason', 'left_px', 'right_px', 'width_px', 'width_diff_m'])

        while True:
            ok_read, frame = cap.read()
            if not ok_read or (max_frames and n >= max_frames):
                break

            img = prepare(frame)
            binary = cv2.bitwise_and(binarize(img, camera, roi=False), roi_mask)
            warped = cv2.warpPerspective(binary, M, (REF_W, REF_H))
            ok, reason, stats = tracker.update(warped)
            accepted += ok
            reasons[reason] += 1
            if stats['width_px'] != "":
                widths.append(stats['width_px'])
                diffs.append(stats['diff_m'])
            log.writerow([n, int(ok), reason, stats['left_px'], stats['right_px'],
                          stats['width_px'], stats['diff_m']])

            smooth = tracker.smoothed()
            if smooth is None:
                out = img.copy()
                put_text(out, ["Lane not detected", f"status: {reason}"])
            else:
                out = draw_lane(img, smooth[0], smooth[1], Minv)
                radius, offset = lane_metrics(smooth[0], smooth[1], tracker.xm)
                radius_txt = "straight" if radius > 3000 else f"{radius:.0f} m"
                side = "left" if offset < 0 else "right"
                status = reason if ok else f"holding last lane ({reason})"
                put_text(out, [f"Radius of Curvature: {radius_txt}",
                               f"Vehicle is {abs(offset):.2f}m {side} of center",
                               f"status: {status}"])
            writer.write(out)
            n += 1
            if n % 200 == 0:
                print(f"  {n} frames")

    cap.release()
    writer.release()

    print(f"{name}: {n} frames, accepted {accepted} ({100 * accepted / max(n, 1):.0f}%) -> {out_path}")
    print("  reasons:", dict(reasons))
    if widths:
        print(f"  raw lane width (both lines found): median {np.median(widths):.0f} px, "
              f"median top-bottom difference {np.median(np.abs(diffs)):.2f} m")
    print(f"  log: {log_path}")


def check_perspective(path, frame_idx):
    os.makedirs(VIDEO_OUT, exist_ok=True)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print("Error: could not read that frame")
        return

    camera = camera_of(frame.shape[1], frame.shape[0])
    img = prepare(frame)
    M = cv2.getPerspectiveTransform(SRC[camera], DST)
    warped = cv2.warpPerspective(img, M, (REF_W, REF_H))

    before = img.copy()
    cv2.polylines(before, [SRC[camera].astype(np.int32)], True, (0, 0, 255), 3)
    after = warped.copy()
    cv2.polylines(after, [DST.astype(np.int32)], True, (0, 0, 255), 3)

    pair = np.hstack([cv2.resize(before, (640, 360)), cv2.resize(after, (640, 360))])
    name = os.path.splitext(os.path.basename(path))[0]
    out_path = f'{VIDEO_OUT}/check_{name}.jpg'
    cv2.imwrite(out_path, pair)
    print(f"Saved {out_path} (camera {camera})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help="path to a video or 'all'")
    parser.add_argument('--check', action='store_true', help="save the perspective check image only")
    parser.add_argument('--frame', type=int, default=0, help="frame used by --check")
    parser.add_argument('--max-frames', type=int, default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(f'{VIDEO_DIR}/*.mp4')) if args.video == 'all' else [args.video]
    for path in paths:
        if args.check:
            check_perspective(path, args.frame)
        else:
            process_video(path, args.max_frames)


if __name__ == "__main__":
    main()