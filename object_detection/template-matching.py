import cv2
import numpy as np

cap = cv2.VideoCapture(1)
template = None
captured = False

print('Press SPACE to capture template, ESC to exit.')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    display = frame.copy()
    # Draw contours before template is captured
    if not captured:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                cv2.drawContours(display, [cnt], -1, (0,255,0), 2)
        cv2.putText(display, 'Press SPACE to capture template', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Frame', display)
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                template = thresh[y:y+h, x:x+w]
                captured = True
                cv2.imshow('Template', template)
            else:
                print('No contour found to capture as template.')
        continue  # Wait for template capture

    # After template is captured, do matching/counting
    count = 0
    if template is not None:
        res = cv2.matchTemplate(thresh, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        loc = np.where(res >= threshold)
        detected = set()
        for pt in zip(*loc[::-1]):
            if all(np.linalg.norm(np.array(pt) - np.array(d)) > min(template.shape)//2 for d in detected):
                detected.add(pt)
                count += 1
                cv2.rectangle(display, pt, (pt[0]+template.shape[1], pt[1]+template.shape[0]), (0,255,0), 2)

    cv2.putText(display, f'Count: {count}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.imshow('Frame', display)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
