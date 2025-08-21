import cv2

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 1000    # adjust for your object size (pixels)
params.maxArea = 100000   # adjust as needed
params.filterByCircularity = False
params.minCircularity = 0.7  # adjust for roundness
params.filterByConvexity = False
params.minConvexity = 0.8
params.filterByInertia = False
params.minInertiaRatio = 0.5

detector = cv2.SimpleBlobDetector_create(params)

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Detect blobs
    keypoints = detector.detect(thresh)
    count = len(keypoints)

    # Draw detected blobs as red circles
    frame_with_keypoints = cv2.drawKeypoints(frame, keypoints, None, (0,0,255),
                                             cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(frame_with_keypoints, f'Count: {count}', (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Blob Detection', frame_with_keypoints)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()