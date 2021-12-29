import cv2
import mediapipe as mp
import time


class FaceMeshDetector:

    def __init__(self, static_mode=False, max_faces=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_faces = max_faces
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.drawingSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=1)
        self.faceMeshDetection = self.mpFaceMesh.FaceMesh(
            static_image_mode=static_mode,
            max_num_faces=max_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence)

    def findFaceMeshes(self, image, draw=True):
        self.output = self.faceMeshDetection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        if self.output.multi_face_landmarks:
            for faceLdmarks in self.output.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, faceLdmarks, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawingSpec, self.drawingSpec)

                face = []
                for id, ldmark in enumerate(faceLdmarks.landmark):
                    h, w, c = image.shape
                    face.append((int(ldmark.x * w), int(ldmark.y * h)))
                faces.append(face)
        return image, faces


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()

    prevTime = 0
    while True:
        res, img = cap.read()
        img, faces = detector.findFaceMeshes(img)
        if faces:
            print(len(faces))

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, "{:.2f}".format(fps), (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        cv2.imshow("Image", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    main()
