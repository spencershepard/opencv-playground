import cv2
import numpy as np
import glob
import os

# Settings
CHESSBOARD_SIZE = (9, 6)
SQUARE_SIZE = 0.8  # Chessboard square size in cm
IMG_DIR = 'calib_images'

# Prepare object points
objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

images = glob.glob(os.path.join(IMG_DIR, '*.png'))

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, None)
    if found:
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, found)
        cv2.imshow('Chessboard', img)
        cv2.waitKey(100)
    else:
        print(f'Chessboard not found in {fname}')

cv2.destroyAllWindows()

if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('Calibration RMS error:', ret)
    print('Camera matrix:\n', mtx)
    print('Distortion coefficients:\n', dist)
    # Save calibration results
    np.savez('calibration_data.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
    print('Calibration data saved to calibration_data.npz')
else:
    print('Not enough valid images for calibration.')
