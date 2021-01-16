import cv2
import pandas as pd

from fellbeast.object_tracker import BoundingBox


class FaceDetector(object):
    face_cascade = cv2.CascadeClassifier('/Users/amitwolfenfeld/Development/fellbeast/fellbeast/models/haarcascade_frontalface_default.xml')
    deepface_detector = cv2.dnn.readNetFromCaffe("/Users/amitwolfenfeld/Development/fellbeast/fellbeast/models/deploy.prototxt",
                                                 "/Users/amitwolfenfeld/Development/fellbeast/fellbeast/models/res10_300x300_ssd_iter_140000.caffemodel")

    def detect(self, image, method='deepface'):
        if method == 'deepface':
            original_size = image.shape
            target_size = (300, 300)
            resized_image = cv2.resize(image, target_size)
            imageBlob = cv2.dnn.blobFromImage(image=resized_image)
            self.deepface_detector.setInput(imageBlob)
            detections = self.deepface_detector.forward()
            column_labels = ["img_id", "is_face", "confidence", "left", "top", "right", "bottom"]
            detections_df = pd.DataFrame(detections[0][0], columns=column_labels)
            detections_df = detections_df[detections_df['is_face'] == 1]
            detections_df = detections_df[detections_df['confidence'] > 0.9]
            detections_df['left'] = (detections_df['left']*original_size[1]).astype(int)
            detections_df['bottom'] = (detections_df['bottom']*original_size[0]).astype(int)
            detections_df['right'] = (detections_df['right']*original_size[1]).astype(int)
            detections_df['top'] = (detections_df['top']*original_size[0]).astype(int)
            detections = detections_df[['top', 'left', 'bottom', 'right']].values.astype(int)

            face_bounding_boxes = [BoundingBox(*detection) for detection in detections]

        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detections = self.face_cascade.detectMultiScale(gray)
            face_bounding_boxes = [BoundingBox.from_tracker(*detection) for detection in detections]
        return face_bounding_boxes

    def trace_faces(self, image, detection_method='deepface'):
        face_bounding_boxes = self.detect(image, method=detection_method)

        for face_bounding_box in face_bounding_boxes:
            top_left, bottom_right = face_bounding_box.rectangle_coordinates

            cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
        return image

    def crop_faces(self, image):
        face_bounding_boxes = self.detect(image, method='deepface')
        if len(face_bounding_boxes) == 0:
            return []
        return [face_bounding_box.crop_image(image) for face_bounding_box in face_bounding_boxes]
