import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

prevTime = 0
currTime = 0

while True:
    res, frame = capture.read()
    rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = hands.process(rgbImage)

    h, w, c = frame.shape

    if output.multi_hand_landmarks:
        for handLandmarks in output.multi_hand_landmarks:
            #for id, landmark in enumerate(handLandmarks.landmark):
                #pass
            px, py = int(handLandmarks.landmark[8].x * w), int(handLandmarks.landmark[8].y * h)
            cv2.circle(frame, (px, py), 17, (255, 0, 255), cv2.FILLED)
            mpDraw.draw_landmarks(frame, handLandmarks, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime

    cv2.putText(frame, "{:.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
