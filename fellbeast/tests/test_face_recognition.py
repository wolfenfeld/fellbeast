import unittest
from PIL import Image

from fellbeast.face_recognition import FaceRecognition


class FaceRecognitionTestCase(unittest.TestCase):
    def test_preprocessing(self):
        face_recognition = FaceRecognition(1.5)
        self.assertEqual(True, False)

    def test_image_comparison(self):
        image_1 = Image.open('./data/yael_1.jpg')
        image_2 = Image.open('./data/amit_1.jpg')
        image_3 = Image.open('./data/yael_3.jpg')
        face_recognition = FaceRecognition(1.5)

        distance = face_recognition.compute_distance(image_1, image_2)
        print(distance)
        distance = face_recognition.compute_distance(image_1, image_3)
        print(distance)


if __name__ == '__main__':
    unittest.main()
