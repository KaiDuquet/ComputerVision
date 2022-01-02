import cv2
import mediapipe as mp
import time


class HandDetector:

    def __init__(self, static_mode=False, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):

        self.static_mode = static_mode
        self.max_hands = max_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpHands = mp.solutions.hands
        self.mpDraw = mp.solutions.drawing_utils

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence)

    def findHands(self, image, draw=True):
        self.output = self.hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.output.multi_hand_landmarks:
            for handLandmarks in self.output.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLandmarks, self.mpHands.HAND_CONNECTIONS)
        return image

    def handCount(self):
        if self.output.multi_hand_landmarks:
            return len(self.output.multi_hand_landmarks)
        return 0

    def getLandmarks(self, dims, handID, draw=True):
        if self.output.multi_hand_landmarks:
            hand = self.output.multi_hand_landmarks[handID]
            self.ldmarks = [(id, int(lm.x * dims[1]), int(lm.y * dims[0])) for id, lm in enumerate(hand.landmark)]
            return self.ldmarks

    def isFingerUp(self, fingerIndex):
        return self.ldmarks[(fingerIndex + 1) * 4][2] < self.ldmarks[(fingerIndex + 1) * 4 - 2][2]


def main():
    prevTime = 0
    currTime = 0

    capture = cv2.VideoCapture(0)
    detector = HandDetector()

    while True:
        res, frame = capture.read()

        frame = detector.detect(frame)
        ldmarks = detector.getLandmarks(frame.shape, 0)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(frame, "{:.2f}".format(fps), (20, 40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
