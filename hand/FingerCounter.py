import cv2
import time
from HandTrackingMod import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

tracker = HandDetector(min_detection_confidence=0.7)


def main():
    prevTime = 0
    while True:
        success, img = cap.read()
        img = tracker.findHands(img)

        ldmarks = tracker.getLandmarks(img.shape, 0)

        if ldmarks:
            fingers = [0, 0, 0, 0, 0]

            fingers[0] = 1 if ldmarks[4][1] > ldmarks[2][1] else 0
            for i in range(1, 5):
                fingers[i] = 1 if ldmarks[(i + 1) * 4][2] < ldmarks[(i + 1) * 4 - 2][2] else 0

            cv2.rectangle(img, (570, 20), (700, 110), (172, 86, 79), cv2.FILLED)
            cv2.putText(img, str(fingers.count(1)), (610, 90), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 2)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, "{:.2f}".format(fps), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Volume Control", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()