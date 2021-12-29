import cv2
import mediapipe as mp
import time


class FaceDetector:

    def __init__(self, min_detection_confidence=0.5):

        self.min_detection_confidence = min_detection_confidence

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils

        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence)

    def findFaces(self, image, draw=True):

        output = self.faceDetection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.output = self.faceDetection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        boxes = []
        if self.output.detections:
            for id, detection in enumerate(self.output.detections):
                unitBox = detection.location_data.relative_bounding_box
                h, w, c = image.shape
                box = int(unitBox.xmin * w), int(unitBox.ymin * h), int(unitBox.width * w), int(unitBox.height * h)
                boxes.append((box, detection.score))
                if draw:
                    cv2.rectangle(image, box, (255, 0, 255), 2)
                    cv2.putText(image, "{:.2f}%".format(detection.score[0] * 100), (box[0], box[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        return image, boxes


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceDetector(0.70)

    prevTime = 0
    while True:
        res, img = cap.read()
        img, boxes = detector.findFaces(img)

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, "{:.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
