import os
import glob
import pickle
import numpy as np
import cv2

nx = 9
ny = 6

objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

objpoints = []
imgpoints = []

images = glob.glob('camera_cal/calibration*.jpg')
if not images:
    images = glob.glob('Zadatak/camera_cal/calibration*.jpg')

print(f"Calibration images found: {len(images)}")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

print(f"Chessboards detected in {len(objpoints)}/{len(images)} images.")

if objpoints:
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera calibration completed successfully.")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)

    with open('calibration.p', 'wb') as f:
        pickle.dump({'mtx': mtx, 'dist': dist}, f)
    print("Saved calibration to 'calibration.p'.")

    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    os.makedirs('examples', exist_ok=True)
    cv2.imwrite('examples/undistort_output.jpg', undistorted)
    print("Saved undistorted test image to 'examples/undistort_output.jpg'.")
else:
    print("Error: No chessboard corners were detected.")