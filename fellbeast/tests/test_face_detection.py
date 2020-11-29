import unittest

import cv2

from fellbeast.face_detection import FaceDetector


class FaceDetectionTestCase(unittest.TestCase):
    def test_face_detection(self):

        image_1 = cv2.imread('./data/yael_1.jpg')

        face_detector = FaceDetector()

        face_detector.trace_faces(image_1)
        cv2.imshow('img', image_1)
        cv2.waitKey()
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
