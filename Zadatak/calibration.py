import numpy as np
import cv2
import glob

# Define inner chessboard corner dimensions
nx = 9
ny = 6

# Prepare 3D object points in real-world coordinates (e.g. (0,0,0), (1,0,0), ...)
objp = np.zeros((nx * ny, 3), np.float32)
objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

# Arrays to store object points and image points from all calibration images
objpoints = []  # 3D points in real-world space
imgpoints = []  # 2D points in image plane

# Load calibration images
images = glob.glob('camera_cal/calibration*.jpg')
if not images:
    images = glob.glob('Zadatak/camera_cal/calibration*.jpg')

print(f"Calibration images found: {len(images)}")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)

if objpoints:
    img = cv2.imread(images[0])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute camera matrix and distortion coefficients
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera calibration completed successfully.")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)

    # Test distortion correction on a sample image
    undistorted = cv2.undistort(img, mtx, dist, None, mtx)
    cv2.imwrite('undistort_output.png', undistorted)
    print("Saved undistorted test image to 'undistort_output.png'.")
else:
    print("Error: No chessboard corners were detected.")