import unittest

import cv2
from PIL import Image

from fellbeast.face_detection import FaceDetector
from fellbeast.face_recognition import FaceRecognition


class FaceRecognitionTestCase(unittest.TestCase):
    def test_preprocessing(self):
        face_recognition = FaceRecognition(1.5)
        self.assertEqual(True, False)

    def test_image_comparison(self):
        image_1 = cv2.imread('./data/yael_1.jpg')
        image_2 = cv2.imread('./data/amit_1.jpg')
        image_3 = cv2.imread('./data/yael_3.jpg')

        face_recognition = FaceRecognition(1.5)
        face_detector = FaceDetector()

        face_image_1 = face_detector.crop_faces(image_1)[0]
        cv2.imshow('yael 1', face_image_1)
        cv2.waitKey(0)
        face_image_2 = face_detector.crop_faces(image_2)[0]
        cv2.imshow('amit 1', face_image_2)
        cv2.waitKey(0)
        face_image_3 = face_detector.crop_faces(image_3)[0]
        cv2.imshow('yael 2', face_image_3)
        cv2.waitKey(0)

        distance = face_recognition.compute_distance(Image.fromarray(face_image_1.astype('uint8'), 'RGB'),
                                                     Image.fromarray(face_image_2.astype('uint8'), 'RGB'))
        print(distance)

        distance = face_recognition.compute_distance(Image.fromarray(face_image_1.astype('uint8'), 'RGB'),
                                                     Image.fromarray(face_image_3.astype('uint8'), 'RGB'))
        print(distance)


if __name__ == '__main__':
    unittest.main()
