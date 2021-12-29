import cv2
import time
import numpy as np
import HandTrackingMod as htm

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

tracker = htm.HandDetector(min_detection_confidence=0.7)

prevTime = 0
while True:
    success, img = cap.read()
    img = tracker.findHands(img)

    if tracker.handCount() == 2:
        ldmarks0 = tracker.getLandmarks(img.shape, 0)
        ldmarks1 = tracker.getLandmarks(img.shape, 1)
        for i in range(4, 21, 4):
            cv2.line(img, ldmarks0[i][1:], ldmarks0[i % 20 + 4][1:], (255, 0, 0), 2)
            cv2.line(img, ldmarks1[i][1:], ldmarks1[i % 20 + 4][1:], (255, 0, 0), 2)
            cv2.line(img, ldmarks0[i][1:], ldmarks1[i][1:], (255, 0, 255), 2)

    currTime = time.time()
    fps = 1 / (currTime - prevTime)
    prevTime = currTime

    cv2.putText(img, "{:.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.imshow("Volume Control", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
