import cv2

from fellbeast.face_detection import FaceDetector
from fellbeast.face_recognition import FaceRecognition


class Camera(object):

    face_recognition = FaceRecognition(0.1, known_face_path='./data/')
    face_detector = FaceDetector()

    def __init__(self, drone_camera, width=320, height=240):
        self.drone_camera = drone_camera
        self.width = width
        self.height = height

    def get_resized_frame(self):
        frame_read = self.drone_camera.get_frame_read()
        resized_frame = cv2.resize(frame_read.frame, (self.width, self.height))
        return resized_frame

    def detect_face(self):
        resized_frame = self.get_resized_frame()
        resized_frame = self.face_detector.trace_faces(resized_frame)
        return resized_frame

    def recognize_faces(self, faces):
        names = self.face_recognition.match_faces_with_names(faces)
        return names
