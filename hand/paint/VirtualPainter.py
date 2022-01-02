import cv2
import time
import os
import numpy as np
from hand.HandTrackingMod import HandDetector


def main():
    overlayImage = cv2.imread('sidebar880.png')
    canvas = np.zeros((720, 1280, 3), np.uint8)
    boundingBoxes = [
        (216, 4, 80), (0, 0, 255),
        (314, 4, 80), (255, 0, 0),
        (408, 4, 80), (0, 255, 255),
        (504, 4, 80), (0, 255, 0),
        (600, 4, 80), (64, 64, 64),
        (700, 4, 80), (255, 255, 255),
        (894, 4, 80), (0, 0, 0)
    ]
    detector = HandDetector(min_detection_confidence=0.85)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    currentDrawColor = (0, 0, 255)

    prevTime = 0
    prevX, prevY = 0, 0
    brushThickness = 15
    while True:
        success, image = cap.read()
        image = cv2.flip(image, 1)

        image = detector.findHands(image)
        ldmarks = detector.getLandmarks(image.shape, 0)

        if ldmarks:
            indexX, indexY = ldmarks[8][1:]
            middleX, middleY = ldmarks[12][1:]

            if detector.isFingerUp(1):
                if detector.isFingerUp(2):
                    prevY, prevX = 0, 0
                    cv2.rectangle(image, (indexX, indexY - 25), (middleX, middleY + 25), currentDrawColor, cv2.FILLED)
                    for i in range(0, len(boundingBoxes), 2):
                        if boundingBoxes[i][1] < indexY < boundingBoxes[i][1] + boundingBoxes[i][2] and \
                                boundingBoxes[i][0] < indexX < boundingBoxes[i][0] + boundingBoxes[i][2]:
                            currentDrawColor = boundingBoxes[i + 1]
                else:
                    if prevX == 0 and prevY == 0:
                        prevX, prevY = indexX, indexY
                    cv2.circle(image, (indexX, indexY), brushThickness, currentDrawColor, cv2.FILLED)
                    cv2.line(canvas, (prevX, prevY), (indexX, indexY), currentDrawColor,
                             80 if currentDrawColor == (0, 0, 0) else brushThickness)
                    prevX, prevY = indexX, indexY

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        imageGrayscale = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imageInverted = cv2.threshold(imageGrayscale, 50, 255, cv2.THRESH_BINARY_INV)
        imageInverted = cv2.cvtColor(imageInverted, cv2.COLOR_GRAY2BGR)
        image = cv2.bitwise_and(image, imageInverted)
        image = cv2.bitwise_or(image, canvas)

        image[0:88, 200:1080] = overlayImage

        cv2.imshow("Virtual Painter", image)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
