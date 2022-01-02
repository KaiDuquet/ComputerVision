import cv2
import mediapipe as mp
import time

mpFace = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFace.FaceDetection(min_detection_confidence=0.70)


def main():
    cap = cv2.VideoCapture(0)

    prevTime = 0
    currTime = 0
    while True:
        res, img = cap.read()
        img = cv2.flip(img, 1)
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = faceDetection.process(imgRGB)

        if output.detections:
            for id, detection in enumerate(output.detections):
                unitBox = detection.location_data.relative_bounding_box
                h, w, c = img.shape
                box = int(unitBox.xmin * w), int(unitBox.ymin * h), int(unitBox.width * w), int(unitBox.height * h)
                cv2.rectangle(img, box, (255, 0, 255), 2)
                cv2.putText(img, "{:.2f}%".format(detection.score[0] * 100), (box[0], box[1] - 20),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, "{:.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
