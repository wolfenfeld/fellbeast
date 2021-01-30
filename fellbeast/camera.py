import cv2
from djitellopy import Tello

from fellbeast.face_detection import FaceDetector
from fellbeast.face_recognition import FaceRecognition


class Camera(object):

    def __init__(self, drone_camera, width=640, height=480, known_face_path='./face_db/'):
        self.drone_camera = drone_camera
        self.width = width
        self.height = height
        self.face_detector = FaceDetector()
        self.face_recognition = FaceRecognition(known_face_path=known_face_path, face_detector=self.face_detector)

    def read(self):
        if type(self.drone_camera) == Tello:
            frame = self.drone_camera.get_frame_read().frame
            return cv2.resize(frame, (self.width, self.height))
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

    def release(self):
        if type(self.drone_camera) == Tello:
            self.drone_camera.cap.release()
        else:
            self.drone_camera.release()
