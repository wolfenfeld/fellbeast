import cv2
from cv2.data import haarcascades


class FaceDetector(object):
    face_cascade = cv2.CascadeClassifier(haarcascades+'haarcascade_frontalface_default.xml')

    def detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.face_cascade.detectMultiScale(gray, 1.1, 4)

    def trace_faces(self, image):
        faces = self.detect(image)

        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        return image

    def crop_faces(self, image):
        faces = self.detect(image)
        return [image[y:y+h, x:x+w] for (x, y, w, h) in faces]
