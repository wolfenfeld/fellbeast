import cv2
from djitellopy import Tello

from fellbeast.face_detection import FaceDetector
from fellbeast.face_recognition import FaceRecognition


class Camera(object):

    face_detector = FaceDetector()
    face_recognition = FaceRecognition(known_face_path='./data/', face_detector=face_detector)

    def __init__(self, drone_camera, width=640, height=480):
        self.drone_camera = drone_camera
        self.width = width
        self.height = height

    def read(self):
        if type(self.drone_camera) == Tello:
            return self.drone_camera.get_frame_read()
        else:
            frame = self.drone_camera.read()[1]
            if frame is None:
                return None
            return cv2.resize(frame, (self.width, self.height))

    def get_resized_frame(self):
        frame_read = self.read()
        resized_frame = cv2.resize(frame_read.frame, (self.width, self.height))
        return resized_frame

    def detect_face(self):
        resized_frame = self.get_resized_frame()
        resized_frame = self.face_detector.trace_faces(resized_frame)
        return resized_frame

