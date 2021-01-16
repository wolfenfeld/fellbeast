import cv2
import face_recognition
import os

SMALLEST_DISTANCE_THRESHOLD = 2
NO_RECOGNITION = SMALLEST_DISTANCE_THRESHOLD + 1


class FaceRecognition(object):

    def __init__(self, face_detector, known_face_path):
        self.face_detector = face_detector
        self.known_face_path = known_face_path
        self.encoded_known_faces = self.get_encoded_known_faces()

    def get_encoded_known_faces(self):
        encoded_known_faces = {}
        for face_file_name in os.listdir(self.known_face_path):
            if face_file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image = cv2.imread(self.known_face_path + face_file_name)
                # Assuming that there is a single face in the known faces images per file
                faces_bounding_boxes = self.face_detector.detect(image)
                if not faces_bounding_boxes:
                    continue
                face_bounding_box = faces_bounding_boxes[0]

                encoded_known_faces[face_file_name] = face_recognition.face_encodings(
                    image, known_face_locations=[face_bounding_box.css])

        return encoded_known_faces

    def find_face_in_encodings(self, image, face_bounding_box):
        encoded_face = face_recognition.face_encodings(image, known_face_locations=[face_bounding_box.css])
        scores = {name: face_recognition.face_distance([encoded_face[0]], encoded_known_face[0])
                  for name, encoded_known_face in self.encoded_known_faces.items()}
        smallest_distance = min(scores.values())
        if smallest_distance < SMALLEST_DISTANCE_THRESHOLD:
            matched_name = min(scores, key=scores.get)
        else:
            matched_name = 'unknown'
        return matched_name

