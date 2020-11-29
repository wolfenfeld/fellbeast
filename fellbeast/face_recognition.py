from res_facenet.models import model_920, model_921

import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


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

    def __init__(self, threshold):
        self.threshold = threshold

    def compute_distance(self, image_1, image_2):
        transformed_image_1 = self.preprocess(image_1).unsqueeze(0)
        transformed_image_2 = self.preprocess(image_2).unsqueeze(0)
        embedded_image_1 = self.model920(transformed_image_1)
        embedded_image_2 = self.model920(transformed_image_2)
        return F.pairwise_distance(embedded_image_1, embedded_image_2)
