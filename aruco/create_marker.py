import cv2
import cv2.aruco as aruco
import argparse

# Parameters
MARKER_ID = 0
DICT_TYPE = aruco.DICT_4X4_50
MARKER_SIZE = 200  # pixels
OUTPUT_FILE = 'aruco_marker.png'

# Create dictionary and marker
aruco_dict = aruco.getPredefinedDictionary(DICT_TYPE)
marker_img = aruco.generateImageMarker(aruco_dict, MARKER_ID, MARKER_SIZE)

# Add white border
BORDER_SIZE = 40  # pixels
bordered_img = cv2.copyMakeBorder(
	marker_img,
	BORDER_SIZE, BORDER_SIZE, BORDER_SIZE, BORDER_SIZE,
	cv2.BORDER_CONSTANT, value=255
)

cv2.imwrite(OUTPUT_FILE, bordered_img)
print(f'Marker saved as {OUTPUT_FILE}')
