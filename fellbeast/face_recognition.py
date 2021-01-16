import cv2
import face_recognition
from res_facenet.models import model_920, model_921

import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image

SMALLEST_DISTANCE_THRESHOLD = 2
NO_RECOGNITION = SMALLEST_DISTANCE_THRESHOLD + 1


class FaceRecognition(object):
    model920 = model_920()
    model921 = model_921()

    # prepare preprocess pipeline
    preprocess_pipelines = [transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                 std=[0.229, 0.224, 0.225])]

    preprocess = transforms.Compose(preprocess_pipelines)

    def __init__(self, threshold, known_face_path, face_detector):
        self.face_detector = face_detector
        self.threshold = threshold
        self.known_face_path = known_face_path
        self.known_faces = self.get_known_faces()
        self.encoded_known_faces = self.get_encoded_known_faces()

    def compute_distance(self, image_1, image_2):

        if not Image.isImageType(image_1):
            image_1 = Image.fromarray(image_1)
        if not Image.isImageType(image_2):
            image_2 = Image.fromarray(image_2)

        transformed_image_1 = self.preprocess(image_1).unsqueeze(0)
        transformed_image_2 = self.preprocess(image_2).unsqueeze(0)
        embedded_image_1 = self.model921(transformed_image_1)
        embedded_image_2 = self.model921(transformed_image_2)
        return F.pairwise_distance(embedded_image_1, embedded_image_2)

    def get_known_faces(self):
        known_faces = {}
        for face_file_name in os.listdir(self.known_face_path):
            if face_file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image = cv2.imread(self.known_face_path+face_file_name)
                # Assuming that there is a single face in the known faces images per file
                face = self.face_detector.crop_faces(image)
                if not face:
                    continue
                known_faces[face_file_name] = face[0]

        return known_faces

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

    def match_faces_with_names(self, faces):
        return [self.match_face_with_known_faces(face) for face in faces]

    def match_face_with_known_faces(self, face):
        scores = {name: self.match_face(face, known_face) for name, known_face in self.known_faces.items()}
        smallest_distance = min(scores.values())
        if smallest_distance < SMALLEST_DISTANCE_THRESHOLD:
            matched_name = min(scores, key=scores.get)
        else:
            matched_name = 'unknown'
        return matched_name

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

