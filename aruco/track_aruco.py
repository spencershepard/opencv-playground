import cv2
import cv2.aruco as aruco
import numpy as np
import os

# Parameters
DICT_TYPE = aruco.DICT_4X4_50
MARKER_LENGTH = 0.025  # Marker size in meters (adjust to your marker)
CAM_ID = 1

# Load camera calibration if available
try:
    # get the path of this script
    cwd = os.path.dirname(os.path.abspath(__file__))
    calibration_path = os.path.join(cwd, '..','calibration', 'calibration_data.npz')
    calib = np.load(calibration_path)
    camera_matrix = calib['mtx']
    dist_coeffs = calib['dist']
    print('Loaded camera calibration data.')
except Exception:
    camera_matrix = None
    dist_coeffs = None
    print('No calibration data found. Pose estimation may be inaccurate.')

aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)
parameters = aruco.DetectorParameters()
# Increase precision by tuning parameters
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX  # Use subpixel refinement
parameters.minDistanceToBorder = 5  # Increase minimum distance to image border
parameters.polygonalApproxAccuracyRate = 0.03  # Lower for stricter polygonal approximation
parameters.adaptiveThreshWinSizeMin = 5
parameters.adaptiveThreshWinSizeMax = 23
parameters.adaptiveThreshWinSizeStep = 4
parameters.adaptiveThreshConstant = 7
parameters.minMarkerPerimeterRate = 0.03  # Lower for smaller markers
parameters.maxMarkerPerimeterRate = 4.0   # Higher for larger markers
parameters.errorCorrectionRate = 0.6      # Higher for more robust detection

cap = cv2.VideoCapture(CAM_ID)

while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to grab frame')
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # print(f'Detected marker IDs: {ids.flatten()}')
        aruco.drawDetectedMarkers(frame, corners, ids)
        if camera_matrix is not None and dist_coeffs is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, MARKER_LENGTH, camera_matrix, dist_coeffs)
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], MARKER_LENGTH/2)
                # Calculate tilt (rotation) relative to marker
                rvec = rvecs[i]
                rot_mat, _ = cv2.Rodrigues(rvec)
                # X/Y tilt: rotation around marker's X/Y axes
                x_tilt = np.degrees(np.arctan2(rot_mat[2,1], rot_mat[2,2]))
                y_tilt = np.degrees(np.arctan2(-rot_mat[2,0], np.sqrt(rot_mat[2,1]**2 + rot_mat[2,2]**2)))
                cv2.putText(frame, f'X tilt: {x_tilt:.1f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.putText(frame, f'Y tilt: {y_tilt:.1f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    # Draw rejected candidates for debugging
    # if rejected is not None and len(rejected) > 0:
    #     aruco.drawDetectedMarkers(frame, rejected, borderColor=(0,0,255))
    cv2.imshow('Aruco Tracker', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
