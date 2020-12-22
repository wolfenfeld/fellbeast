import cv2
from res_facenet.models import model_920, model_921

import torch.nn.functional as F
from torchvision import transforms
import os
from PIL import Image

# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
from fellbeast.face_detection import FaceDetector


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
    face_detector = FaceDetector()

    def __init__(self, threshold, known_face_path):
        self.threshold = threshold
        self.known_face_path = known_face_path
        self.known_faces = self.get_known_faces()

    def compute_distance(self, image_1, image_2):

        if not Image.isImageType(image_1):
            image_1 = Image.fromarray(image_1)
        if not Image.isImageType(image_2):
            image_2 = Image.fromarray(image_2)

        transformed_image_1 = self.preprocess(image_1).unsqueeze(0)
        transformed_image_2 = self.preprocess(image_2).unsqueeze(0)
        embedded_image_1 = self.model920(transformed_image_1)
        embedded_image_2 = self.model920(transformed_image_2)
        return F.pairwise_distance(embedded_image_1, embedded_image_2)

    def get_known_faces(self):
        known_faces = {}
        for face_file_name in os.listdir(self.known_face_path):
            image = cv2.imread(self.known_face_path+face_file_name)
            # Assuming that there is a single face in the known faces images per file
            known_faces[face_file_name] = self.face_detector.crop_faces(image)[0]

        return known_faces

    def match_faces_with_names(self, faces):
        return [self.match_face_with_known_faces(face) for face in faces]

    def match_face_with_known_faces(self, face):
        best_match_score = -1
        matched_name = "None"
        for name, known_face in self.known_faces.items():
            match_score = self.compute_distance(face, known_face)
            if match_score > best_match_score:
                matched_name = name
                best_match_score = match_score

        return matched_name
