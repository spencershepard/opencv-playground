import cv2
import os
import numpy as np

# Settings
CHESSBOARD_SIZE = (9,6)
SAVE_DIR = 'calib_images'
CAM_ID = 1

os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(CAM_ID)
img_count = 0

print('Press SPACE to capture image, ESC to exit.')

while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to grab frame')
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_SIZE, None)

    display = frame.copy()
    if found:
        cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
        cv2.putText(display, 'Chessboard Found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    else:
        cv2.putText(display, 'Not Found', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('Camera', display)
    key = cv2.waitKey(1)


    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        if found:
            filename = os.path.join(SAVE_DIR, f'chessboard_{img_count:02d}.png')
            cv2.imwrite(filename, frame)
            print(f'Saved {filename}')
            img_count += 1
        else:
            print('Chessboard not found, image not saved.')

cap.release()
cv2.destroyAllWindows()
